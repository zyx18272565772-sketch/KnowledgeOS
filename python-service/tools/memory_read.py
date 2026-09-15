from typing import Dict, Any, List
from tools.base import Tool, ToolSchema, SchemaProperty, ToolMetadata
from core.config import config
from core.redis_client import redis_client
from core.llm import LLMService

# 压缩阈值配置（按用户发起的一轮问答计数，而不是按单条消息计数）
COMPRESS_THRESHOLD = 10  # 超过 10 轮对话触发压缩
KEEP_RECENT = 5          # 保留最近 5 轮完整对话


class ConversationMemoryReadTool(Tool):
    """对话记忆读取工具（含上下文压缩）"""

    def __init__(self):
        input_schema = ToolSchema(
            properties={
                "conversation_id": SchemaProperty(
                    type="string",
                    description="对话ID",
                    required=True
                ),
                "limit": SchemaProperty(
                    type="number",
                    description="返回消息数量限制",
                    required=False,
                    default=10
                )
            },
            type="object"
        )

        output_schema = ToolSchema(
            properties={
                "messages": SchemaProperty(
                    type="array",
                    description="对话消息列表",
                    required=True
                ),
                "conversation_id": SchemaProperty(
                    type="string",
                    description="对话ID",
                    required=True
                ),
                "total_count": SchemaProperty(
                    type="number",
                    description="总消息数量",
                    required=True
                ),
                "compressed": SchemaProperty(
                    type="boolean",
                    description="是否已压缩",
                    required=False
                ),
                "summary_generated": SchemaProperty(
                    type="boolean",
                    description="是否成功生成历史摘要",
                    required=False
                )
            },
            type="object"
        )

        metadata = ToolMetadata(
            timeout_ms=2000,  # 快速失败，避免阻塞等待
            max_retries=0,
            permission="user",
            description="读取对话记忆"
        )

        super().__init__(
            name="conversation_memory_read",
            description="读取对话记忆",
            input_schema=input_schema,
            output_schema=output_schema,
            metadata=metadata
        )

        self.llm_service = LLMService()

    def execute(self, parameters: Dict[str, Any]) -> Dict[str, Any]:
        """执行对话记忆读取（含上下文压缩）"""
        conversation_id = parameters.get("conversation_id")
        limit = int(parameters.get("limit", 10))

        config.logger.info(f"Reading conversation memory for ID: {conversation_id}")

        # 按用户消息计算“轮数”：一条 user 消息代表一轮对话的开始，
        # assistant 的回复属于同一轮。
        all_messages = redis_client.get_all_messages(conversation_id)
        total_count = len(all_messages)
        round_count = self._count_rounds(all_messages)

        if total_count == 0:
            #没有消息，返回空列表
            return {
                "messages": [],
                "conversation_id": conversation_id,
                "total_count": 0,
                "round_count": 0,
                "compressed": False,
                "summary_generated": False
            }

        if round_count <= COMPRESS_THRESHOLD:
            # 消息不多，直接返回最近的
            messages = redis_client.get_messages(conversation_id, limit)
            return {
                "messages": messages,
                "conversation_id": conversation_id,
                "total_count": total_count,
                "round_count": round_count,
                "compressed": False,
                "summary_generated": False
            }

        #超过轮数阈值，压缩早期历史并保留最近 N 轮完整对话。
        recent_start = self._recent_round_start(all_messages, KEEP_RECENT)
        early_messages = all_messages[:recent_start]
        recent_messages = all_messages[recent_start:]

        # 摘要游标记录缓存摘要已覆盖到第几条消息；之后只处理新增早期历史。
        cached_summary = redis_client.get_summary(conversation_id)
        cached_cursor = redis_client.get_summary_cursor(conversation_id)
        if cached_summary and self._is_legacy_fallback_summary(cached_summary):
            cached_summary = None
            cached_cursor = None
        summary = None

        target_cursor = recent_start
        has_valid_cached_state = (
            cached_summary is not None
            and cached_cursor is not None
            and 0 <= cached_cursor <= target_cursor
        )

        if has_valid_cached_state and cached_cursor == target_cursor:
            summary = cached_summary
            config.logger.info(f"Using cached summary for conversation {conversation_id}")
        else:
            try:
                if has_valid_cached_state:
                    incremental_messages = all_messages[cached_cursor:target_cursor]
                    summary = self._compress_history(
                        incremental_messages,
                        conversation_id,
                        existing_summary=cached_summary,
                    )
                    config.logger.info(
                        "Incrementally compressed %s messages for conversation %s",
                        len(incremental_messages),
                        conversation_id,
                    )
                else:
                    summary = self._compress_history(early_messages, conversation_id)
                    config.logger.info(
                        "Created summary from %s messages for conversation %s",
                        len(early_messages),
                        conversation_id,
                    )

                redis_client.set_summary_state(
                    conversation_id,
                    summary,
                    target_cursor,
                )
            except Exception as e:
                config.logger.warning(f"Failed to generate conversation summary: {e}")

        # 返回：摘要 + 最近5轮
        messages = recent_messages
        if summary:
            messages = [
                {"role": "system", "content": f"[历史对话摘要] {summary}"}
            ] + recent_messages

        return {
            "messages": messages,
            "conversation_id": conversation_id,
            "total_count": total_count,
            "round_count": round_count,
            "compressed": True,
            "summary_generated": bool(summary),
            "original_count": total_count
        }

    @staticmethod
    def _count_rounds(messages: List[Dict]) -> int:
        """按用户消息计算对话轮数。"""
        return sum(1 for message in messages if message.get("role") == "user")

    @staticmethod
    def _recent_round_start(messages: List[Dict], keep_rounds: int) -> int:
        """返回最近 keep_rounds 轮的起始位置，保留用户消息与其后的回复。"""
        user_indexes = [
            index for index, message in enumerate(messages)
            if message.get("role") == "user"
        ]
        if len(user_indexes) <= keep_rounds:
            return 0
        return user_indexes[-keep_rounds]

    @staticmethod
    def _is_legacy_fallback_summary(summary: str) -> bool:
        return (
            summary.startswith("用户进行了 ")
            and summary.endswith(" 轮对话，讨论了相关知识问题。")
        )

    def _compress_history(self, messages: List[Dict], conversation_id: str,
                          existing_summary: str = "") -> str:
        """用 LLM 创建或增量更新历史对话摘要。"""
        if not messages:
            return existing_summary

        history_text = "\n".join([
            f"{m.get('role', 'unknown')}: {m.get('content', '')}"
            for m in messages
        ])

        if existing_summary:
            prompt = f"""请更新一段会话历史摘要。

已有摘要：
{existing_summary}

新增的早期对话：
{history_text}

请将两部分合并为新的 2-3 句话摘要。保留用户问题、已确认结论、用户偏好、未完成任务；删除重复和过时信息。只输出更新后的摘要，不要解释。"""
        else:
            prompt = f"""请将以下对话压缩成简短摘要，保留关键信息：
1. 用户问了什么问题
2. AI 回答了什么要点
3. 用户的偏好或关注点
4. 尚未完成的任务

对话内容：
{history_text}

请用 2-3 句话概括，只输出摘要，不要解释。"""

        try:
            summary = self.llm_service.chat(prompt)
            if not summary:
                raise RuntimeError("LLM returned an empty conversation summary")
            return summary
        except Exception as e:
            config.logger.warning(f"Failed to compress history: {e}")
            raise
