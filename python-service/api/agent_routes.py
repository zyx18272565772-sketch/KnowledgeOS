from fastapi import APIRouter, Depends
from fastapi.responses import StreamingResponse
from pydantic import BaseModel, Field
from workflows import RouterAgent
from typing import List, Optional
from collections import defaultdict
from api.auth import get_current_user, require_admin
from core.redis_client import redis_client
import time
import logging
import json

logger = logging.getLogger(__name__)

router = APIRouter(dependencies=[Depends(get_current_user)])

router_agent = RouterAgent()


class AttachmentInput(BaseModel):
    filename: str = Field(..., min_length=1, max_length=255)
    file_type: str = "document"
    content: str = Field(..., min_length=1, max_length=50000)
    chunks_count: int = 0
    char_count: int = 0


class AgentRunRequest(BaseModel):
    input: str
    conversation_id: Optional[str] = None
    user_id: Optional[str] = None
    context: Optional[str] = ""
    attachments: List[AttachmentInput] = Field(default_factory=list)
    goal: Optional[str] = None
    run_id: Optional[str] = None
    trace_id: Optional[str] = None
    stream: bool = False


task_stats = defaultdict(lambda: {
    "total": 0,
    "success": 0,
    "failed": 0,
    "total_duration_ms": 0
})


@router.post("/agent/run/stream")
async def run_agent_stream(request: AgentRunRequest, current_user: dict = Depends(get_current_user)):
    """
    流式执行Agent (Server-Sent Events)

    Args:
        request: Agent运行请求

    Returns:
        SSE事件流
    """
    async def event_generator():
        start_time = time.time()
        task_type = "unknown"
        has_error = False

        if request.conversation_id and not request.conversation_id.startswith(f"{current_user['username']}_"):
            yield f"data: {json.dumps({'type': 'error', 'content': '无权访问该会话'})}\n\n"
            return

        if len(request.attachments) > 5:
            yield f"data: {json.dumps({'type': 'error', 'content': '单次最多上传5个附件'})}\n\n"
            return

        try:
            logger.info(f"[Agent接口] 收到流式请求：{request.input[:50]}...")
            attachments = [attachment.model_dump() for attachment in request.attachments]
            if request.conversation_id:
                try:
                    if attachments:
                        # 原图已由 parse-temp 删除；这里只缓存解析后的文字和元数据。
                        redis_client.set_attachments(request.conversation_id, attachments)
                    else:
                        attachments = redis_client.get_attachments(request.conversation_id)
                        for attachment in attachments:
                            attachment["_from_cache"] = True
                except Exception as cache_error:
                    # 附件缓存属于会话增强能力，Redis 短暂异常不应阻断当前问答。
                    logger.warning(
                        "会话 %s 的附件缓存暂时不可用：%s",
                        request.conversation_id,
                        cache_error,
                    )

            # 路由分类可能需要一次 LLM 调用；先把等待状态推给前端，避免界面无反馈。
            yield f"data: {json.dumps({'type': 'routing_started'})}\n\n"

            for event_data in router_agent.route_stream(
                input_text=request.input,
                conversation_id=request.conversation_id,
                user_id=current_user["username"],
                is_admin=bool(current_user.get("is_admin")),
                context=request.context or "",
                attachments=attachments,
                goal=request.goal,
                run_id=request.run_id,
                trace_id=request.trace_id
            ):
                parsed = json.loads(event_data.strip().replace("data: ", "").replace("\n\n", ""))
                if parsed.get("type") == "routed":
                    task_type = parsed.get("task_type", "unknown")
                elif parsed.get("type") in {"error", "step_failed"}:
                    has_error = True

                yield f"data: {event_data}\n\n"

            duration_ms = (time.time() - start_time) * 1000
            task_stats[task_type]["total"] += 1
            if has_error:
                task_stats[task_type]["failed"] += 1
            else:
                task_stats[task_type]["success"] += 1
            task_stats[task_type]["total_duration_ms"] += duration_ms

            yield f"data: {json.dumps({'type': 'complete', 'success': not has_error})}\n\n"

        except Exception as e:
            duration_ms = (time.time() - start_time) * 1000
            task_stats[task_type]["total"] += 1
            task_stats[task_type]["failed"] += 1
            task_stats[task_type]["total_duration_ms"] += duration_ms

            logger.error(f"[Agent接口] 流式请求处理失败：{str(e)}")
            yield f"data: {json.dumps({'type': 'error', 'content': str(e)})}\n\n"

    return StreamingResponse(
        event_generator(),
        media_type="text/event-stream",
        headers={
            "Cache-Control": "no-cache",
            "Connection": "keep-alive",
            "X-Accel-Buffering": "no"
        }
    )


@router.get("/agent/stats")
async def get_agent_stats(_current_user: dict = Depends(require_admin)):
    """
    获取Agent任务统计信息

    Returns:
        任务类型统计
    """
    stats = []
    total_all = 0
    success_all = 0
    failed_all = 0

    for task_type, data in task_stats.items():
        total = data["total"]
        success = data["success"]
        failed = data["failed"]
        avg_duration = data["total_duration_ms"] / total if total > 0 else 0

        total_all += total
        success_all += success
        failed_all += failed

        stats.append({
            "task_type": task_type,
            "total": total,
            "success": success,
            "failed": failed,
            "success_rate": round(success / total * 100, 2) if total > 0 else 0,
            "avg_duration_ms": round(avg_duration, 2)
        })

    return {
        "task_stats": stats,
        "summary": {
            "total": total_all,
            "success": success_all,
            "failed": failed_all,
            "overall_success_rate": round(success_all / total_all * 100, 2) if total_all > 0 else 0
        },
        "keyword_stats": router_agent.get_task_stats()
    }


@router.get("/agent/report")
async def get_agent_report(_current_user: dict = Depends(require_admin)):
    """
    自动报表统计（管理后台用）

    Returns:
        文档数、chunk数、Agent运行统计、热门问题等
    """
    # Agent 运行统计（复用 task_stats）
    stats = []
    total_all = 0
    success_all = 0
    failed_all = 0
    for task_type, data in task_stats.items():
        total = data["total"]
        success = data["success"]
        failed = data["failed"]
        total_all += total
        success_all += success
        failed_all += failed
        stats.append({
            "task_type": task_type,
            "total": total,
            "success": success,
            "failed": failed,
            "success_rate": round(success / total * 100, 2) if total > 0 else 0,
        })

    # 文档 / 向量库统计
    doc_count = 0
    chunk_count = 0
    try:
        from core.vector_store import vector_store
        vs = vector_store.get_stats()
        chunk_count = vs.get("doc_count", 0)
    except Exception:
        pass
    try:
        from core.mysql_client import mysql_client
        mysql_client._ensure_connected()
        conn = mysql_client.connection
        with conn.cursor() as cur:
            cur.execute("SELECT COUNT(*) FROM documents")
            doc_count = cur.fetchone()[0]
    except Exception:
        doc_count = 0

    return {
        "summary": {
            "doc_count": doc_count,
            "chunk_count": chunk_count,
            "agent_runs": total_all,
            "success_count": success_all,
            "failed_count": failed_all,
            "success_rate": round(success_all / total_all * 100, 2) if total_all > 0 else 0,
        },
        "task_stats": stats,
    }

