from fastapi import APIRouter, BackgroundTasks, Depends, UploadFile, File, Form, HTTPException
from starlette.concurrency import run_in_threadpool
from typing import Optional
from pydantic import BaseModel
from langchain_core.documents import Document
from core.parser import DocumentParser
from core.vector_store import vector_store
from core.mysql_client import mysql_client
from api.auth import get_current_user, require_admin
from workflows.inspection_agent import InspectionAgent
import os
import re
import time
import logging
import json
import threading

logger = logging.getLogger(__name__)

router = APIRouter(dependencies=[Depends(get_current_user)])
parser = DocumentParser()

UPLOAD_DIR = os.getenv("UPLOAD_DIR", "./uploads")
os.makedirs(UPLOAD_DIR, exist_ok=True)

TEMP_DOCUMENT_EXTENSIONS = {".pdf", ".docx", ".txt", ".md"}
TEMP_IMAGE_EXTENSIONS = {".png", ".jpg", ".jpeg", ".gif", ".bmp", ".tiff", ".tif"}
MAX_TEMP_UPLOAD_BYTES = int(os.getenv("MAX_TEMP_UPLOAD_MB", "10")) * 1024 * 1024


_table_ready = False
_publish_lock = threading.Lock()


class RejectRequest(BaseModel):
    reason: Optional[str] = None


def _ensure_tables():
    """确保所需的表存在，延迟初始化，失败不阻塞启动"""
    global _table_ready
    if _table_ready:
        return
    try:
        mysql_client.execute("""
            CREATE TABLE IF NOT EXISTS documents (
                id INT AUTO_INCREMENT PRIMARY KEY,
                title VARCHAR(255) NOT NULL,
                filename VARCHAR(255) NOT NULL,
                file_path VARCHAR(512),
                file_size INT DEFAULT 0,
                file_type VARCHAR(50),
                status VARCHAR(20) DEFAULT 'processing',
                chunks_count INT DEFAULT 0,
                error_msg TEXT,
                create_time DATETIME DEFAULT CURRENT_TIMESTAMP,
                update_time DATETIME DEFAULT CURRENT_TIMESTAMP ON UPDATE CURRENT_TIMESTAMP
            ) ENGINE=InnoDB DEFAULT CHARSET=utf8mb4
        """)
        mysql_client.execute("""
            CREATE TABLE IF NOT EXISTS knowledge_chunk (
                id INT AUTO_INCREMENT PRIMARY KEY,
                doc_id INT NOT NULL,
                chunk_index INT NOT NULL,
                chunk_text LONGTEXT NOT NULL,
                create_time DATETIME DEFAULT CURRENT_TIMESTAMP,
                UNIQUE KEY uk_document_chunk (doc_id, chunk_index),
                INDEX idx_knowledge_chunk_doc_id (doc_id)
            ) ENGINE=InnoDB DEFAULT CHARSET=utf8mb4
        """)
        mysql_client.execute("""
            CREATE TABLE IF NOT EXISTS document_inspections (
                document_id INT PRIMARY KEY,
                inspection_status VARCHAR(20) NOT NULL DEFAULT 'PENDING',
                inspection_time DATETIME NULL,
                low_quality_count INT NOT NULL DEFAULT 0,
                duplicate_count INT NOT NULL DEFAULT 0,
                conflict_count INT NOT NULL DEFAULT 0,
                report_json LONGTEXT,
                error_message TEXT,
                reject_reason TEXT,
                update_time DATETIME DEFAULT CURRENT_TIMESTAMP ON UPDATE CURRENT_TIMESTAMP,
                INDEX idx_document_inspection_status (inspection_status)
            ) ENGINE=InnoDB DEFAULT CHARSET=utf8mb4
        """)
        # 兼容升级前已完成且已经存在于正式 FAISS 中的文档。
        mysql_client.execute(
            "UPDATE documents SET status='ACTIVE' WHERE status IN ('completed', 'COMPLETED')"
        )
        _table_ready = True
        logger.info("文档表、知识片段表和巡检结果表已就绪")
    except Exception as e:
        logger.warning(f"文档相关数据表初始化失败：{e}")
        raise RuntimeError("文档数据表初始化失败") from e


def _document_objects(document: dict, chunk_rows: list) -> list:
    """把 MySQL 中的片段恢复成 LangChain Document，供巡检/发布复用。"""
    source = document.get("filename") or document.get("title") or "未知来源"
    return [
        Document(
            page_content=str(row.get("chunk_text") or ""),
            metadata={
                "doc_id": int(document["id"]),
                "source": source,
                "chunk_index": row.get("chunk_index", index),
                "status": "ACTIVE",
            },
        )
        for index, row in enumerate(chunk_rows)
    ]


def _run_pending_inspection(document_id: int) -> None:
    """上传响应返回后自动执行预发布巡检；异常已在 Agent 内持久化。"""
    try:
        InspectionAgent().inspect_pending_document(document_id)
    except Exception:
        # InspectionAgent 已记录 FAILED；这里不能自动发布，也不重复打印全文。
        logger.warning("文档 %s 自动巡检失败，已保持待审核", document_id)


def _approve_document(document_id: int) -> dict:
    """串行发布并在数据库更新失败时尽量补偿正式 FAISS。"""
    with _publish_lock:
        document = mysql_client.get_document(document_id)
        if not document:
            raise HTTPException(status_code=404, detail="文档不存在")
        if document.get("status") == "ACTIVE":
            # 幂等返回，防止前端重试造成重复向量。
            return {"status": "success", "document_id": document_id, "document_status": "ACTIVE"}
        if document.get("status") != "PENDING_REVIEW":
            raise HTTPException(status_code=409, detail="只有待审核文档可以发布")

        inspection = mysql_client.get_inspection(document_id)
        if not inspection or inspection.get("inspection_status") != "COMPLETED":
            raise HTTPException(status_code=409, detail="巡检尚未成功完成，不能发布")

        chunks = mysql_client.get_chunks_by_doc(document_id)
        if not chunks:
            raise HTTPException(status_code=409, detail="文档没有可发布的知识片段")

        if vector_store.contains_document(document_id):
            # 处理上一次“向量已落盘、状态更新响应丢失”的极端情况。
            mysql_client.update_document_status(document_id, "ACTIVE", len(chunks))
            vector_store.delete_pending_embeddings(document_id)
            logger.warning("文档 %s 已存在于正式 FAISS，本次仅修复为 ACTIVE", document_id)
            return {"status": "success", "document_id": document_id, "document_status": "ACTIVE"}

        vectors = vector_store.load_pending_embeddings(document_id)
        documents = _document_objects(document, chunks)
        try:
            vector_store.publish_precomputed_documents(documents, vectors)
            mysql_client.update_document_status(document_id, "ACTIVE", len(chunks))
        except HTTPException:
            raise
        except Exception:
            # 操作顺序是先向量、后状态；状态写入失败时删除刚写入的 doc_id。
            try:
                vector_store.delete_document(document_id)
            except Exception as cleanup_error:
                logger.error("文档 %s 发布补偿失败：%s", document_id, cleanup_error)
            raise

        vector_store.delete_pending_embeddings(document_id)
        logger.info("文档 %s 已审核通过并正式发布到 FAISS", document_id)
        return {
            "status": "success",
            "document_id": document_id,
            "document_status": "ACTIVE",
            "chunks_count": len(chunks),
        }


@router.post("/documents/upload")
async def upload_document(
    background_tasks: BackgroundTasks,
    file: UploadFile = File(...),
    title: Optional[str] = Form(None),
    _current_user: dict = Depends(require_admin),
):
    """上传、解析、分块并生成待审核向量；不会直接进入线上 FAISS。"""
    _ensure_tables()
    start_time = time.time()

    if not file.filename:
        raise HTTPException(status_code=400, detail="文件名不能为空")

    file_ext = os.path.splitext(file.filename)[1]
    safe_filename = f"{int(time.time())}_{file.filename}"
    file_path = os.path.join(UPLOAD_DIR, safe_filename)

    content = await file.read()
    with open(file_path, "wb") as f:
        f.write(content)

    file_size = len(content)
    doc_id = None

    try:
        doc_title = title or os.path.splitext(file.filename)[0]
        doc_id = mysql_client.create_document(
            title=doc_title,
            filename=file.filename,
            file_path=file_path,
            file_size=file_size,
            file_type=file_ext.lstrip("."),
        )

        if not doc_id:
            raise HTTPException(status_code=500, detail="创建文档记录失败")

        chunks = await run_in_threadpool(parser.parse, file_path)
        if not chunks:
            raise ValueError("未能从文档中解析出可用文本")
        for chunk in chunks:
            chunk.metadata["doc_id"] = doc_id
            chunk.metadata["source"] = file.filename

        inserted_count = mysql_client.insert_chunks(doc_id, [
            {
                "page_content": c.page_content,
                "chunk_index": c.metadata.get("chunk_index", i),
            }
            for i, c in enumerate(chunks)
        ])
        if inserted_count != len(chunks):
            raise RuntimeError(f"知识片段写入不完整：预期 {len(chunks)} 条，实际 {inserted_count} 条")

        mysql_client.update_document_status(doc_id, "PENDING_REVIEW", len(chunks))
        mysql_client.upsert_inspection(doc_id, "PENDING")

        try:
            await run_in_threadpool(vector_store.create_pending_embeddings, doc_id, chunks)
        except Exception as embedding_error:
            mysql_client.upsert_inspection(
                doc_id,
                "FAILED",
                error_message=f"Embedding 失败：{str(embedding_error)[:900]}",
            )
            logger.error("文档 %s 的待审核向量生成失败：%s", doc_id, embedding_error)
            return {
                "status": "success",
                "doc_id": doc_id,
                "filename": file.filename,
                "document_status": "PENDING_REVIEW",
                "inspection_status": "FAILED",
                "chunks_count": len(chunks),
                "message": "文档已上传，但向量生成失败；请在待审核列表中重新巡检。",
            }

        background_tasks.add_task(_run_pending_inspection, doc_id)

        process_time = time.time() - start_time
        logger.info(
            "文档上传完成：document_id=%s，文件=%s，片段=%s，已进入待审核，耗时=%.2f秒",
            doc_id,
            file.filename,
            len(chunks),
            process_time,
        )

        return {
            "status": "success",
            "doc_id": doc_id,
            "filename": file.filename,
            "document_status": "PENDING_REVIEW",
            "inspection_status": "PENDING",
            "chunks_count": len(chunks),
            "process_time": round(process_time, 2),
            "message": "文档上传成功，正在进行知识巡检。",
        }

    except HTTPException:
        raise
    except Exception as e:
        if doc_id:
            try:
                vector_store.delete_pending_embeddings(doc_id)
            except Exception as cleanup_error:
                logger.warning(f"Failed to clean up vectors for document {doc_id}: {cleanup_error}")
            try:
                mysql_client.delete_chunks_by_doc(doc_id)
                mysql_client.update_document_status(doc_id, "PROCESSING_FAILED", 0, str(e))
            except Exception:
                pass
        logger.error(f"文档上传处理失败：{str(e)}", exc_info=True)
        raise HTTPException(status_code=500, detail=f"文档处理失败: {str(e)}")


@router.get("/documents")
async def list_documents(page: int = 1, page_size: int = 20):
    """获取文档列表"""
    _ensure_tables()
    try:
        documents = mysql_client.list_documents(page, page_size)
        total = mysql_client.get_document_count()
        stats = mysql_client.get_document_stats()
        return {
            "documents": documents or [],
            "total": total,
            "page": page,
            "page_size": page_size,
            "stats": stats,
        }
    except Exception as e:
        logger.error(f"获取文档列表失败：{str(e)}")
        raise HTTPException(status_code=500, detail=f"获取文档列表失败: {str(e)}")


@router.get("/documents/{doc_id}/inspection")
async def get_document_inspection(
    doc_id: int,
    _current_user: dict = Depends(require_admin),
):
    """查看一份文档持久化的巡检报告。"""
    _ensure_tables()
    document = mysql_client.get_document(doc_id)
    if not document:
        raise HTTPException(status_code=404, detail="文档不存在")
    inspection = mysql_client.get_inspection(doc_id)
    return {"document": document, "inspection": inspection}


@router.post("/documents/{doc_id}/inspect")
async def reinspect_document(
    doc_id: int,
    _current_user: dict = Depends(require_admin),
):
    """管理员手动重新巡检待审核文档。"""
    _ensure_tables()
    document = mysql_client.get_document(doc_id)
    if not document:
        raise HTTPException(status_code=404, detail="文档不存在")
    if document.get("status") != "PENDING_REVIEW":
        raise HTTPException(status_code=409, detail="只有待审核文档可以重新巡检")
    chunks = mysql_client.get_chunks_by_doc(doc_id)
    documents = _document_objects(document, chunks)
    try:
        vector_store.load_pending_embeddings(doc_id)
    except (FileNotFoundError, ValueError):
        await run_in_threadpool(vector_store.create_pending_embeddings, doc_id, documents)
    try:
        report = await run_in_threadpool(InspectionAgent().inspect_pending_document, doc_id)
        judge_complete = bool(report.get("summary", {}).get("judge_complete", True))
        return {
            "status": "success" if judge_complete else "failed",
            "inspection_status": "COMPLETED" if judge_complete else "FAILED",
            "document_id": doc_id,
            "report": report,
        }
    except Exception as exc:
        raise HTTPException(status_code=500, detail=f"巡检失败，文档仍保持待审核：{exc}")


@router.post("/documents/{doc_id}/approve")
async def approve_document(
    doc_id: int,
    _current_user: dict = Depends(require_admin),
):
    """审核通过后才将预计算向量写入正式 FAISS。"""
    _ensure_tables()
    try:
        return await run_in_threadpool(_approve_document, doc_id)
    except HTTPException:
        raise
    except Exception as exc:
        logger.error("文档 %s 发布失败，仍保持待审核：%s", doc_id, exc, exc_info=True)
        raise HTTPException(status_code=500, detail=f"发布失败，文档仍保持待审核：{exc}")


@router.post("/documents/{doc_id}/reject")
async def reject_document(
    doc_id: int,
    payload: RejectRequest,
    _current_user: dict = Depends(require_admin),
):
    """驳回待审核文档，确保其永远不进入正式 FAISS。"""
    _ensure_tables()
    document = mysql_client.get_document(doc_id)
    if not document:
        raise HTTPException(status_code=404, detail="文档不存在")
    if document.get("status") != "PENDING_REVIEW":
        raise HTTPException(status_code=409, detail="只有待审核文档可以驳回")
    mysql_client.update_document_status(
        doc_id,
        "REJECTED",
        int(document.get("chunks_count") or 0),
    )
    mysql_client.reject_inspection(doc_id, (payload.reason or "").strip()[:500] or None)
    vector_store.delete_pending_embeddings(doc_id)
    logger.info("文档 %s 已由管理员驳回，未写入正式 FAISS", doc_id)
    return {"status": "success", "document_id": doc_id, "document_status": "REJECTED"}


@router.delete("/documents/{doc_id}")
async def delete_document(doc_id: int, _current_user: dict = Depends(require_admin)):
    """删除文档（向量 + MySQL + 文件）"""
    doc = mysql_client.get_document(doc_id)
    if not doc:
        raise HTTPException(status_code=404, detail="文档不存在")

    try:
        if doc.get("status") == "ACTIVE":
            vector_store.delete_document(doc_id)
        vector_store.delete_pending_embeddings(doc_id)
    except Exception as e:
        logger.warning(f"Failed to delete vectors for doc {doc_id}: {e}")

    mysql_client.delete_inspection(doc_id)
    mysql_client.delete_chunks_by_doc(doc_id)
    mysql_client.delete_document(doc_id)

    file_path = doc.get("file_path")
    if file_path and os.path.exists(file_path):
        try:
            os.remove(file_path)
        except OSError:
            logger.warning(f"Failed to delete file: {file_path}")

    return {"status": "success", "message": f"文档 {doc_id} 已删除"}


@router.get("/conversations")
async def list_conversations(current_user: dict = Depends(get_current_user)):
    """获取对话列表，可按用户过滤"""
    try:
        from core.redis_client import redis_client
        username = current_user["username"]
        pattern = f"conversation:{username}_*:messages"
        keys = redis_client.client.keys(pattern)
        conversations = []
        for key in keys:
            # key format: conversation:{username}_{conv_id}:messages
            parts = key.split(":")
            conv_id = parts[1] if len(parts) > 1 else ""
            if username and not conv_id.startswith(username + "_"):
                continue
            msg_count = redis_client.get_message_count(conv_id)
            summary = redis_client.get_summary(conv_id)
            conversations.append({
                "conversation_id": conv_id,
                "message_count": msg_count,
                "summary": summary or "未命名对话",
            })
        conversations.sort(key=lambda x: x.get("message_count", 0), reverse=True)
        return {"conversations": conversations}
    except Exception as e:
        logger.error(f"Error listing conversations: {str(e)}")
        import traceback
        logger.error(traceback.format_exc())
        raise HTTPException(status_code=500, detail=f"获取对话列表失败: {str(e)}")


@router.get("/conversations/{conversation_id}")
async def get_conversation(conversation_id: str, current_user: dict = Depends(get_current_user)):
    """获取对话消息"""
    try:
        if not conversation_id.startswith(f"{current_user['username']}_"):
            raise HTTPException(status_code=403, detail="无权访问该会话")
        from core.redis_client import redis_client
        messages = redis_client.get_all_messages(conversation_id)
        return {"conversation_id": conversation_id, "messages": messages or []}
    except HTTPException:
        raise
    except Exception as e:
        logger.error(f"Error getting conversation: {str(e)}")
        raise HTTPException(status_code=500, detail="获取对话消息失败")


@router.delete("/conversations/{conversation_id}")
async def delete_conversation(conversation_id: str, current_user: dict = Depends(get_current_user)):
    """删除对话（清空 Redis 中的消息和摘要）"""
    try:
        if not conversation_id.startswith(f"{current_user['username']}_"):
            raise HTTPException(status_code=403, detail="无权访问该会话")
        from core.redis_client import redis_client
        redis_client.clear_conversation(conversation_id)
        logger.info(f"Conversation deleted: {conversation_id}")
        return {"ok": True, "conversation_id": conversation_id}
    except HTTPException:
        raise
    except Exception as e:
        logger.error(f"Error deleting conversation: {str(e)}")
        raise HTTPException(status_code=500, detail="删除对话失败")


@router.post("/documents/parse-temp")
async def parse_temp_document(
    file: UploadFile = File(...),
):
    """解析文件并返回文本内容，不存入知识库，仅用于当前对话"""
    if not file.filename:
        raise HTTPException(status_code=400, detail="文件名不能为空")

    file_ext = os.path.splitext(file.filename)[1].lower()
    allowed_extensions = TEMP_DOCUMENT_EXTENSIONS | TEMP_IMAGE_EXTENSIONS
    if file_ext not in allowed_extensions:
        raise HTTPException(
            status_code=400,
            detail="仅支持 PDF、DOCX、TXT、MD 和常见图片格式",
        )

    original_filename = os.path.basename(file.filename)
    sanitized_filename = re.sub(r'[^0-9A-Za-z._\-\u4e00-\u9fff]', '_', original_filename)
    safe_filename = f"temp_{int(time.time())}_{sanitized_filename}"
    file_path = os.path.join(UPLOAD_DIR, safe_filename)

    content = await file.read(MAX_TEMP_UPLOAD_BYTES + 1)
    if len(content) > MAX_TEMP_UPLOAD_BYTES:
        max_mb = MAX_TEMP_UPLOAD_BYTES // (1024 * 1024)
        raise HTTPException(status_code=413, detail=f"附件不能超过{max_mb}MB")

    with open(file_path, "wb") as f:
        f.write(content)

    try:
        # 文档解析和 OCR 都是同步阻塞操作，放入线程池避免阻塞 FastAPI 事件循环。
        chunks = await run_in_threadpool(parser.parse, file_path)
        text = "\n\n".join([c.page_content for c in chunks]).strip()
        char_count = len(re.sub(r'\s+', '', text))

        if file_ext in TEMP_IMAGE_EXTENSIONS:
            ocr_errors = [
                chunk.metadata.get("error")
                for chunk in chunks
                if getattr(chunk, "metadata", {}).get("error")
            ]
            failed_markers = ("图片OCR处理失败", "图片中未识别到文字")
            if ocr_errors or any(marker in text for marker in failed_markers):
                detail = ocr_errors[0] if ocr_errors else "图片中未识别到文字"
                raise HTTPException(status_code=422, detail=f"OCR识别失败：{detail}")
            if char_count < 4:
                raise HTTPException(status_code=422, detail="OCR识别文字过少，请上传更清晰的截图")

        if not text:
            raise HTTPException(status_code=422, detail="附件中未解析到有效文字")

        file_type = "image" if file_ext in TEMP_IMAGE_EXTENSIONS else "document"
        return {
            "status": "success",
            "filename": original_filename,
            "file_type": file_type,
            "content": text,
            "text": text,
            "char_count": char_count,
            "chunks_count": len(chunks),
        }
    except HTTPException:
        raise
    except Exception as e:
        logger.error(f"Temp parse failed: {str(e)}", exc_info=True)
        raise HTTPException(status_code=500, detail=f"解析失败: {str(e)}")
    finally:
        if os.path.exists(file_path):
            try:
                os.remove(file_path)
            except OSError:
                pass


