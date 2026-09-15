import redis
import json
import time
import logging
from core.config import config

logger = logging.getLogger(__name__)

class RedisClient:
    """Redis 客户端，用于会话记忆存储"""

    def __init__(self):
        """初始化Redis连接池"""
        self.pool = redis.ConnectionPool(
            host=config.REDIS_HOST,
            port=config.REDIS_PORT,
            password=config.REDIS_PASSWORD if config.REDIS_PASSWORD else None,
            db=config.REDIS_DB,
            decode_responses=True,
            max_connections=20,
            protocol=2,  # RESP2 兼容旧版 Redis
        )
        self.client = redis.Redis(connection_pool=self.pool)
        logger.info(f"Redis 客户端初始化完成（RESP2）：{config.REDIS_HOST}:{config.REDIS_PORT}")

    def add_message(self, conversation_id: str, role: str, content: str):
        """追加消息到会话"""
        key = f"conversation:{conversation_id}:messages"
        message = json.dumps({
            "role": role,
            "content": content,
            "timestamp": time.time()
        }, ensure_ascii=False)
        self.client.rpush(key, message)
        # 不设过期时间，永久保存
        logger.debug(f"消息已加入会话 {conversation_id}：角色={role}")

    def get_messages(self, conversation_id: str, limit: int = 10) -> list:
        """获取最近 N 条消息"""
        key = f"conversation:{conversation_id}:messages"
        messages = self.client.lrange(key, -limit, -1)
        return [json.loads(m) for m in messages]

    def get_all_messages(self, conversation_id: str) -> list:
        """获取所有消息"""
        key = f"conversation:{conversation_id}:messages"
        messages = self.client.lrange(key, 0, -1)
        return [json.loads(m) for m in messages]

    def get_message_count(self, conversation_id: str) -> int:
        """获取消息总数"""
        key = f"conversation:{conversation_id}:messages"
        return self.client.llen(key)

    def clear_conversation(self, conversation_id: str):
        """清空会话"""
        key = f"conversation:{conversation_id}:messages"
        summary_key = f"conversation:{conversation_id}:summary"
        summary_cursor_key = f"conversation:{conversation_id}:summary_cursor"
        attachments_key = f"conversation:{conversation_id}:attachments"
        self.client.delete(key)
        self.client.delete(summary_key)
        self.client.delete(summary_cursor_key)
        self.client.delete(attachments_key)
        logger.info(f"会话 {conversation_id} 已清空")

    def set_attachments(self, conversation_id: str, attachments: list,
                        expire: int = None):
        """按会话临时缓存附件解析结果；这里只保存 OCR/文档文字，不保存原文件。"""
        key = f"conversation:{conversation_id}:attachments"
        ttl = expire if expire is not None else config.ATTACHMENT_CACHE_TTL
        payload = json.dumps(attachments, ensure_ascii=False)
        self.client.setex(key, ttl, payload)
        logger.debug(
            "已为会话 %s 缓存解析后的附件，数量=%s，过期时间=%s秒",
            conversation_id, len(attachments), ttl,
        )

    def get_attachments(self, conversation_id: str) -> list:
        """读取当前会话尚未过期的附件解析结果。"""
        key = f"conversation:{conversation_id}:attachments"
        payload = self.client.get(key)
        if not payload:
            return []
        try:
            attachments = json.loads(payload)
            return attachments if isinstance(attachments, list) else []
        except (TypeError, ValueError, json.JSONDecodeError):
            logger.warning("会话 %s 的附件缓存格式无效，已清除", conversation_id)
            self.client.delete(key)
            return []

    def clear_attachments(self, conversation_id: str):
        """只清除会话附件缓存，不影响聊天消息。"""
        self.client.delete(f"conversation:{conversation_id}:attachments")

    def get_summary(self, conversation_id: str) -> str:
        """获取会话摘要"""
        summary_key = f"conversation:{conversation_id}:summary"
        return self.client.get(summary_key)

    def set_summary(self, conversation_id: str, summary: str, expire: int = 3600):
        """设置会话摘要"""
        summary_key = f"conversation:{conversation_id}:summary"
        self.client.setex(summary_key, expire, summary)
        logger.debug(f"会话 {conversation_id} 的摘要已写入")

    def get_summary_cursor(self, conversation_id: str):
        """获取已被摘要覆盖的消息数量。"""
        cursor_key = f"conversation:{conversation_id}:summary_cursor"
        value = self.client.get(cursor_key)
        try:
            return int(value) if value is not None else None
        except (TypeError, ValueError):
            return None

    def set_summary_state(self, conversation_id: str, summary: str,
                          summarized_message_count: int, expire: int = 3600):
        """原子写入摘要与游标，并让两者使用相同过期时间。"""
        summary_key = f"conversation:{conversation_id}:summary"
        cursor_key = f"conversation:{conversation_id}:summary_cursor"
        with self.client.pipeline() as pipeline:
            pipeline.setex(summary_key, expire, summary)
            pipeline.setex(cursor_key, expire, str(summarized_message_count))
            pipeline.execute()
        logger.debug(
            "会话 %s 的摘要状态已写入，覆盖到第 %s 条消息",
            conversation_id,
            summarized_message_count,
        )

# 创建全局实例
redis_client = RedisClient()
