from typing import Dict, Any, List, Optional, Generator, Tuple
from enum import Enum
from workflows.knowledge_qa_agent import KnowledgeQAAgent
from workflows.chitchat_agent import ChitChatAgent
from intent.classifier import IntentClassifier, IntentResult, IntentType
import logging
import json

logger = logging.getLogger(__name__)


class TaskType(Enum):
    """任务类型枚举"""
    CHITCHAT = "chitchat"
    KNOWLEDGE_QA = "knowledge_qa"
    KNOWLEDGE_INSPECTION = "knowledge_inspection"
    REASONING = "reasoning"
    UNKNOWN = "unknown"


class RouterAgent:
    """路由Agent - 负责将用户请求路由到合适的工作流"""

    def __init__(self):
        self.knowledge_qa_agent = KnowledgeQAAgent()
        self.chitchat_agent = ChitChatAgent()
        self.classifier = IntentClassifier()
        self._reasoning_agent = None

    @property
    def reasoning_agent(self):
        """延迟加载 ReasoningAgent"""
        if self._reasoning_agent is None:
            from workflows.reasoning_agent import ReasoningAgent
            self._reasoning_agent = ReasoningAgent()
        return self._reasoning_agent

    def route_stream(self, input_text: str, conversation_id: Optional[str] = None,
                     user_id: Optional[str] = None, is_admin: bool = False,
                     context: str = "",
                     attachments: Optional[List[Dict[str, Any]]] = None,
                     **kwargs) -> Generator[str, None, None]:
        """
        流式路由并执行任务
        Args:
            input_text: 用户输入
            conversation_id: 会话ID
            user_id: 用户ID
            context: 对话上下文
            **kwargs: 其他参数

        Yields:
            JSON格式的事件流
        """
        attachments = attachments or []
        task_type, intent_result = self.classify_task_with_result(input_text, attachments)
        # 缓存附件只在当前问题确实引用它时进入下游，避免污染无关问答。
        if attachments and not intent_result.attachment_action:
            attachments = []
        complexity = self.classify_complexity(
            input_text,
            has_context=bool(context.strip()),
        )
        task_type_name = {
            TaskType.CHITCHAT: "闲聊",
            TaskType.KNOWLEDGE_QA: "知识问答",
            TaskType.KNOWLEDGE_INSPECTION: "知识巡检",
            TaskType.REASONING: "深度推理",
            TaskType.UNKNOWN: "未知任务",
        }.get(task_type, task_type.value)
        logger.info(f"[路由Agent] 流式路由结果：{task_type_name}")

        try:
            yield json.dumps({
                "type": "routed",
                "task_type": task_type.value,
                "attachment_action": intent_result.attachment_action,
            })

            if task_type == TaskType.CHITCHAT:
                for event in self.chitchat_agent.chat_stream(
                    input_text, conversation_id, user_id, context, **kwargs
                ):
                    yield event

            elif task_type == TaskType.KNOWLEDGE_QA:
                for event in self.knowledge_qa_agent.ask_stream(
                    input_text, conversation_id, user_id, context,
                    complexity=complexity,
                    attachments=attachments,
                    attachment_action=intent_result.attachment_action,
                    needs_knowledge_search=intent_result.needs_knowledge_search,
                    retrieval_query=intent_result.retrieval_query,
                    **kwargs
                ):
                    yield event

            elif task_type == TaskType.REASONING:
                yield json.dumps({"type": "start", "content": ""})
                reasoning_context = context
                attachment_context = self._format_attachment_context(attachments)
                if attachment_context:
                    reasoning_context = (
                        f"{reasoning_context}\n\n{attachment_context}"
                        if reasoning_context else attachment_context
                    )
                for event in self.reasoning_agent.reason_stream(
                    input_text, reasoning_context, conversation_id
                ):
                    yield event

            else:
                for event in self.knowledge_qa_agent.ask_stream(
                    input_text, conversation_id, user_id, context,
                    attachments=attachments,
                    attachment_action=intent_result.attachment_action,
                    needs_knowledge_search=intent_result.needs_knowledge_search,
                    retrieval_query=intent_result.retrieval_query,
                    **kwargs
                ):
                    yield event

        except Exception as e:
            logger.error(f"[路由Agent] 流式路由执行失败：{str(e)}")
            yield json.dumps({
                "type": "error",
                "content": str(e)
            })

    @staticmethod
    def _format_attachment_context(attachments: List[Dict[str, Any]]) -> str:
        parts = []
        for attachment in attachments:
            filename = attachment.get("filename", "未命名附件")
            content = str(attachment.get("content") or attachment.get("text") or "").strip()
            if content:
                parts.append(f"【附件：{filename}】\n{content}")
        return "\n\n".join(parts)[:12000]

    def classify_task_with_result(
        self,
        input_text: str,
        attachments: Optional[List[Dict[str, Any]]] = None,
    ) -> Tuple[TaskType, IntentResult]:
        attachments = attachments or []
        result = (
            self.classifier.classify(input_text, attachments)
            if attachments else self.classifier.classify(input_text)
        )

        intent_to_task = {
            IntentType.CHITCHAT: TaskType.CHITCHAT,
            IntentType.KNOWLEDGE_QA: TaskType.KNOWLEDGE_QA,
            # 知识巡检已迁移到文档审核工作流；旧模型即使返回该枚举，
            # 聊天链路也只按普通知识问答处理。
            IntentType.KNOWLEDGE_INSPECTION: TaskType.KNOWLEDGE_QA,
            IntentType.IDENTITY_QUERY: TaskType.CHITCHAT,
            IntentType.UNKNOWN: TaskType.KNOWLEDGE_QA,
        }
        task_type = intent_to_task.get(result.intent, TaskType.KNOWLEDGE_QA)

        if attachments and result.attachment_action:
            task_type = (
                TaskType.REASONING
                if result.attachment_action == "reasoning"
                else TaskType.KNOWLEDGE_QA
            )
        elif task_type == TaskType.KNOWLEDGE_QA:
            complexity = self.classify_complexity(input_text)
            if complexity == "complex":
                task_type = TaskType.REASONING

        return task_type, result

    def classify_task(self, input_text: str) -> TaskType:
        """
        分类任务类型 - 委托给统一分类器

        Args:
            input_text: 用户输入
        Returns:
            任务类型
        """
        task_type, _ = self.classify_task_with_result(input_text)
        return task_type

    def classify_complexity(self, input_text: str, has_context: bool = False) -> str:
        """
        判断问题复杂度，决定走哪条链路

        Returns:
            "simple"  — L1：直接检索+生成
            "medium" — L2：读记忆/问题改写+检索+生成
            "complex" — L3：分解子问题+逐个推理+汇总
        """
        # L3：显式的对比、分析、归纳请求需要多信息检索与汇总；
        # 不能用字数做门槛，例如“对比 A 和 B 的优缺点”虽短但仍是复杂任务。
        l3_indicators = ["对比", "比较", "优缺点", "区别", "异同",
                         "分析", "总结", "归纳", "评估", "权衡"]
        if any(ind in input_text for ind in l3_indicators):
            return "complex"

        # L2：需要改写/上下文的指标
        l2_indicators = ["它", "这个", "那个", "上面", "之前", "刚才"]
        has_pronoun = any(ind in input_text for ind in l2_indicators)

        if has_context or has_pronoun:
            return "medium"

        # L1：简单问题
        return "simple"

    def _parse_inspection_type(self, input_text: str) -> str:
        """从输入中解析巡检类型"""
        lower_text = input_text.lower()

        if "重复" in lower_text:
            return "duplicate"
        elif "低质量" in lower_text or "质量" in lower_text or "片段" in lower_text:
            return "low_quality"
        elif "过期" in lower_text or "陈旧" in lower_text:
            return "stale"
        else:
            return "full"

    def get_task_stats(self) -> Dict[str, int]:
        """获取各类关键词数量统计（用于调试和分析）"""
        return self.classifier.get_keyword_stats()
