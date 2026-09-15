import os
import json
import mysql.connector
from mysql.connector import Error
from typing import List, Dict, Any, Optional
import logging

logger = logging.getLogger(__name__)

class MySQLClient:
    """MySQL 数据库客户端"""
    
    def __init__(self):
        self.host = os.getenv("MYSQL_HOST", "localhost")
        self.port = int(os.getenv("MYSQL_PORT", "3306"))
        self.database = os.getenv("MYSQL_DATABASE", "ai_knowledge_db")
        self.username = os.getenv("MYSQL_USERNAME", "root")
        self.password = os.getenv("MYSQL_PASSWORD", "")
        self.connection = None
    
    def connect(self):
        """建立数据库连接"""
        try:
            self.connection = mysql.connector.connect(
                host=self.host,
                port=self.port,
                database=self.database,
                user=self.username,
                password=self.password,
                charset='utf8mb4',
                time_zone='+08:00'
            )
            if self.connection.is_connected():
                logger.info("MySQL 数据库连接成功")
                cursor = self.connection.cursor()
                cursor.execute("SET time_zone = '+08:00'")
                cursor.close()
        except Error as e:
            self.connection = None
            logger.error(f"MySQL 数据库连接失败：{e}")

    def _ensure_connected(self):
        """确保连接可用，失败则抛出明确异常"""
        if not self.connection or not self.connection.is_connected():
            self.connect()
        if self.connection is None or not self.connection.is_connected():
            raise RuntimeError(
                f"无法连接到 MySQL ({self.host}:{self.port}/{self.database})，"
                f"请检查 MySQL 是否运行，以及 .env 中的 MYSQL_HOST/MYSQL_PORT/MYSQL_USERNAME/MYSQL_PASSWORD 配置"
            )
    
    def insert_chunks(self, doc_id: int, chunks: List[Dict[str, Any]]):
        """批量插入 chunks 到 knowledge_chunk 表"""
        if not chunks:
            return 0
        
        self._ensure_connected()
        
        try:
            cursor = self.connection.cursor()
            
            # 先删除该文档已有的 chunks（避免重复）
            delete_sql = "DELETE FROM knowledge_chunk WHERE doc_id = %s"
            cursor.execute(delete_sql, (doc_id,))
            
            # 批量插入新 chunks
            insert_sql = """
                INSERT INTO knowledge_chunk (doc_id, chunk_index, chunk_text, create_time)
                VALUES (%s, %s, %s, NOW())
            """
            
            data = [
                (doc_id, chunk.get('chunk_index', i), chunk.get('page_content', ''),)
                for i, chunk in enumerate(chunks)
            ]
            
            cursor.executemany(insert_sql, data)
            self.connection.commit()
            
            inserted_count = cursor.rowcount
            logger.info(f"文档 {doc_id} 已写入 {inserted_count} 个知识片段")
            
            cursor.close()
            return inserted_count
        
        except Error as e:
            logger.error(f"知识片段写入失败：{e}")
            if self.connection:
                self.connection.rollback()
            return 0
    
    def fetch_one(self, sql: str, params: tuple = None) -> Optional[Dict[str, Any]]:
        """执行查询并返回单行结果"""
        self._ensure_connected()
        
        try:
            cursor = self.connection.cursor(dictionary=True)
            
            if params:
                cursor.execute(sql, params)
            else:
                cursor.execute(sql)
            
            result = cursor.fetchone()
            cursor.close()
            return result
        
        except Error as e:
            logger.error(f"MySQL 单条查询失败：{e}")
            raise

    def fetch_all(self, sql: str, params: tuple = None) -> List[Dict[str, Any]]:
        """执行查询并返回所有结果"""
        self._ensure_connected()
        
        try:
            cursor = self.connection.cursor(dictionary=True)
            
            if params:
                cursor.execute(sql, params)
            else:
                cursor.execute(sql)
            
            result = cursor.fetchall()
            cursor.close()
            return result
        
        except Error as e:
            logger.error(f"MySQL 多条查询失败：{e}")
            raise

    def execute(self, sql: str, params: tuple = None) -> int:
        """执行SQL语句（INSERT/UPDATE/DELETE）"""
        self._ensure_connected()
        
        try:
            cursor = self.connection.cursor()
            
            if params:
                cursor.execute(sql, params)
            else:
                cursor.execute(sql)
            
            self.connection.commit()
            affected_rows = cursor.rowcount
            cursor.close()
            return affected_rows
        
        except Error as e:
            logger.error(f"SQL 执行失败：{e}")
            if self.connection:
                self.connection.rollback()
            raise

    def create_document(self, title: str, filename: str,
                        file_path: str, file_size: int,
                        file_type: str) -> int:
        """创建文档记录，返回 doc_id"""
        self._ensure_connected()
        try:
            cursor = self.connection.cursor()
            cursor.execute("""
                INSERT INTO documents (title, filename, file_path, file_size, file_type, status, create_time)
                VALUES (%s, %s, %s, %s, %s, 'processing', NOW())
            """, (title, filename, file_path, file_size, file_type))
            self.connection.commit()
            doc_id = cursor.lastrowid
            cursor.close()
            logger.info(f"文档记录创建成功：ID={doc_id}，标题={title}")
            return doc_id
        except Error as e:
            logger.error(f"文档记录创建失败：{e}")
            if self.connection:
                self.connection.rollback()
            return 0

    def update_document_status(self, doc_id: int, status: str,
                               chunks_count: int = 0, error_msg: str = None):
        """更新文档状态"""
        self._ensure_connected()
        try:
            cursor = self.connection.cursor()
            if error_msg is not None:
                cursor.execute("""
                    UPDATE documents SET status=%s, chunks_count=%s, error_msg=%s
                    WHERE id=%s
                """, (status, chunks_count, error_msg, doc_id))
            else:
                cursor.execute("""
                    UPDATE documents SET status=%s, chunks_count=%s, error_msg=NULL
                    WHERE id=%s
                """, (status, chunks_count, doc_id))
            self.connection.commit()
            cursor.close()
            logger.info(f"文档 {doc_id} 状态已更新为 {status}")
        except Error as e:
            logger.error(f"文档状态更新失败：{e}")
            if self.connection:
                self.connection.rollback()
            raise

    def list_documents(self, page: int = 1, page_size: int = 20) -> List[Dict[str, Any]]:
        """获取文档列表（分页）"""
        offset = (page - 1) * page_size
        rows = self.fetch_all(
            """
            SELECT d.*, i.inspection_status, i.inspection_time,
                   COALESCE(i.low_quality_count, 0) AS low_quality_count,
                   COALESCE(i.duplicate_count, 0) AS duplicate_count,
                   COALESCE(i.conflict_count, 0) AS conflict_count,
                   i.error_message AS inspection_error,
                   i.reject_reason
            FROM documents d
            LEFT JOIN document_inspections i ON i.document_id = d.id
            ORDER BY d.create_time DESC
            LIMIT %s OFFSET %s
            """,
            (page_size, offset)
        )
        return rows

    def get_document_count(self) -> int:
        """获取文档总数"""
        result = self.fetch_one("SELECT COUNT(*) as cnt FROM documents")
        return result["cnt"] if result else 0

    def get_document(self, doc_id: int) -> Dict[str, Any]:
        """获取单个文档"""
        return self.fetch_one("SELECT * FROM documents WHERE id=%s", (doc_id,))

    def delete_document(self, doc_id: int) -> int:
        """删除文档记录"""
        return self.execute("DELETE FROM documents WHERE id=%s", (doc_id,))

    def delete_chunks_by_doc(self, doc_id: int) -> int:
        """删除文档的所有 chunks"""
        return self.execute("DELETE FROM knowledge_chunk WHERE doc_id=%s", (doc_id,))

    def get_chunks_by_doc(self, doc_id: int) -> List[Dict[str, Any]]:
        """按原始顺序获取某份文档的知识片段。"""
        return self.fetch_all(
            """
            SELECT id, doc_id, chunk_index, chunk_text, create_time
            FROM knowledge_chunk
            WHERE doc_id=%s
            ORDER BY chunk_index ASC
            """,
            (doc_id,),
        )

    def get_active_document_ids(self) -> List[int]:
        """返回当前允许进入线上检索的文档 ID。"""
        rows = self.fetch_all("SELECT id FROM documents WHERE status='ACTIVE'")
        return [int(row["id"]) for row in rows]

    def upsert_inspection(
        self,
        doc_id: int,
        inspection_status: str,
        report: Optional[Dict[str, Any]] = None,
        low_quality_count: int = 0,
        duplicate_count: int = 0,
        conflict_count: int = 0,
        error_message: Optional[str] = None,
        reject_reason: Optional[str] = None,
    ) -> int:
        """保存某份文档最新一次巡检状态与报告。"""
        report_json = json.dumps(report, ensure_ascii=False) if report is not None else None
        return self.execute(
            """
            INSERT INTO document_inspections (
                document_id, inspection_status, inspection_time,
                low_quality_count, duplicate_count, conflict_count,
                report_json, error_message, reject_reason, update_time
            ) VALUES (%s, %s, NOW(), %s, %s, %s, %s, %s, %s, NOW())
            ON DUPLICATE KEY UPDATE
                inspection_status=VALUES(inspection_status),
                inspection_time=VALUES(inspection_time),
                low_quality_count=VALUES(low_quality_count),
                duplicate_count=VALUES(duplicate_count),
                conflict_count=VALUES(conflict_count),
                report_json=VALUES(report_json),
                error_message=VALUES(error_message),
                reject_reason=COALESCE(VALUES(reject_reason), reject_reason),
                update_time=NOW()
            """,
            (
                doc_id, inspection_status, low_quality_count,
                duplicate_count, conflict_count, report_json,
                error_message, reject_reason,
            ),
        )

    def get_inspection(self, doc_id: int) -> Optional[Dict[str, Any]]:
        """获取并反序列化某份文档的最新巡检结果。"""
        row = self.fetch_one(
            "SELECT * FROM document_inspections WHERE document_id=%s",
            (doc_id,),
        )
        if not row:
            return None
        raw_report = row.pop("report_json", None)
        try:
            row["report"] = json.loads(raw_report) if raw_report else None
        except (TypeError, ValueError, json.JSONDecodeError):
            row["report"] = None
        return row

    def get_document_stats(self) -> Dict[str, int]:
        """知识库管理页所需的状态统计。"""
        rows = self.fetch_all(
            "SELECT status, COUNT(*) AS cnt FROM documents GROUP BY status"
        )
        counts = {str(row["status"]): int(row["cnt"]) for row in rows}
        conflict = self.fetch_one(
            """
            SELECT COALESCE(SUM(conflict_count), 0) AS cnt
            FROM document_inspections i
            JOIN documents d ON d.id=i.document_id
            WHERE d.status='PENDING_REVIEW'
              AND i.inspection_status='COMPLETED'
            """
        )
        return {
            "active": counts.get("ACTIVE", 0),
            "pending_review": counts.get("PENDING_REVIEW", 0),
            "rejected": counts.get("REJECTED", 0),
            "conflicts": int((conflict or {}).get("cnt") or 0),
        }

    def reject_inspection(self, doc_id: int, reason: Optional[str]) -> int:
        """保留原巡检报告，只记录管理员驳回结果。"""
        return self.execute(
            """
            UPDATE document_inspections
            SET reject_reason=%s, update_time=NOW()
            WHERE document_id=%s
            """,
            (reason, doc_id),
        )

    def delete_inspection(self, doc_id: int) -> int:
        """删除文档时同步清理巡检结果。"""
        return self.execute(
            "DELETE FROM document_inspections WHERE document_id=%s",
            (doc_id,),
        )

# 创建全局实例
mysql_client = MySQLClient()
