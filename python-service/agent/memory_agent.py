"""Memory Agent - 会话记忆管理 Agent"""

from tools.registry import tool_registry
import logging

logger = logging.getLogger(__name__)


class MemoryAgent:
    """负责读取和保存会话历史。"""

    def load_memory(self, state) -> str:
        """加载会话记忆上下文。"""
        context_parts = []

        # 1. 加载会话记忆
        if state.conversation_id and tool_registry.has_tool("conversation_memory_read"):
            try:
                history = tool_registry.invoke_tool(
                    "conversation_memory_read",
                    {
                        "conversation_id": state.conversation_id,
                        "limit": 10
                    }
                )
                messages = history.get("messages", [])
                if messages:
                    formatted = self._format_history(messages)
                    context_parts.append(formatted)
                    logger.info(f"[{state.run_id}] MemoryAgent loaded {len(messages)} messages"
                                f" (compressed: {history.get('compressed', False)})")
            except Exception as e:
                logger.warning(f"[{state.run_id}] MemoryAgent failed to load conversation: {e}")

        return "\n\n".join(context_parts) if context_parts else ""

    def save_memory(self, state, question: str, answer: str):
        """保存会话记忆（用户问题 + AI回答）。"""
        if not state.conversation_id:
            return

        # 1. 写入用户问题
        if tool_registry.has_tool("conversation_memory_write"):
            try:
                tool_registry.invoke_tool(
                    "conversation_memory_write",
                    {
                        "conversation_id": state.conversation_id,
                        "role": "user",
                        "content": question
                    }
                )
            except Exception as e:
                logger.warning(f"[{state.run_id}] MemoryAgent failed to write user message: {e}")

        # 2. 写入 AI 回答
        if tool_registry.has_tool("conversation_memory_write"):
            try:
                tool_registry.invoke_tool(
                    "conversation_memory_write",
                    {
                        "conversation_id": state.conversation_id,
                        "role": "assistant",
                        "content": answer
                    }
                )
            except Exception as e:
                logger.warning(f"[{state.run_id}] MemoryAgent failed to write assistant message: {e}")

    def _format_history(self, messages: list) -> str:
        """格式化对话历史"""
        if not messages:
            return ""

        formatted = []
        for msg in messages:
            role = msg.get("role", "unknown")
            content = msg.get("content", "")
            if role == "system":
                formatted.append(content)
            elif role == "user":
                formatted.append(f"用户: {content}")
            elif role == "assistant":
                formatted.append(f"AI: {content}")
            else:
                formatted.append(content)

        return "\n".join(formatted)
