from typing import Dict, Any, List, Optional, Tuple
from agent.state import AgentState, AgentStep, StepType, TerminationCondition, IntermediateConclusion
from intent.classifier import IntentClassifier, IntentType, IntentResult
from dataclasses import dataclass
import time
import logging

logger = logging.getLogger(__name__)


@dataclass
class RewriteResult:
    """问题改写结果"""
    original_question: str
    rewritten_question: str
    rewrite_type: str


@dataclass
class SufficiencyResult:
    """结果充分性判断"""
    is_sufficient: bool
    reasoning: str
    missing_aspects: List[str]
    suggestions: List[str]


class Planner:
    """任务规划器 - 负责分析任务、规划步骤、判断状态"""

    def __init__(self):
        self.classifier = IntentClassifier()

    def recognize_intent(self, state: AgentState) -> IntentResult:
        """意图识别 - 委托给统一分类器"""
        question = state.original_input or ""
        return (
            self.classifier.classify(question, state.attachments)
            if state.attachments else self.classifier.classify(question)
        )

    def check_clarification_needed(self, state: AgentState, intent: IntentResult) -> Tuple[bool, Optional[str]]:
        """判断是否需要澄清"""
        question = state.original_input or ""

        # 如果有用户上传的文件作为上下文，不需要澄清
        if state.context and state.context.strip():
            return False, None

        if intent.requires_clarification:
            return True, intent.clarification_prompt

        vague_indicators = ["那个", "它", "这个", "他", "她", "这事", "那事"]
        if any(ind in question for ind in vague_indicators) and len(question) < 15:
            return True, "您是指什么？请提供更多具体信息。"

        if len(question) < 5:
            return True, "您的问题太简略了，请详细描述一下您想了解的内容。"

        return False, None

    def rewrite_question(self, question: str, conversation_context: str = "",
                         rewrite_type: str = "semantic") -> RewriteResult:
        """问题改写 - LLM 语义改写，失败时 fallback 到简单替换"""
        if rewrite_type == "simple":
            return self._rewrite_simple(question)

        # 尝试 LLM 语义改写
        llm = self._get_llm()
        if llm:
            try:
                return self._rewrite_with_llm(question, conversation_context, llm)
            except Exception as e:
                logger.warning(f"LLM rewrite failed, falling back to simple: {e}")

        return self._rewrite_simple(question)

    def _get_llm(self):
        """延迟获取 LLM 实例"""
        if not hasattr(self, '_llm'):
            try:
                from core.llm import llm_service
                self._llm = llm_service.llm
            except Exception:
                self._llm = None
        return self._llm

    def _rewrite_with_llm(self, question: str, conversation_context: str, llm) -> RewriteResult:
        """LLM 语义改写"""
        from langchain_core.prompts import PromptTemplate
        from langchain_core.output_parsers import StrOutputParser

        prompt = PromptTemplate.from_template(
            """请对以下用户问题进行改写，目标是提升知识库检索的召回率。

改写规则：
1. 补全省略的主语/宾语
2. 将口语化表达转为书面化
3. 将代词替换为具体指代（结合上下文）
4. 保留原始意图，不要改变问题含义

对话上下文：{conversation_context}

用户问题：{question}

请直接输出改写后的问题，不要解释。"""
        )

        chain = prompt | llm | StrOutputParser()
        rewritten = chain.invoke({
            "question": question,
            "conversation_context": conversation_context or "无"
        }).strip()

        return RewriteResult(
            original_question=question,
            rewritten_question=rewritten,
            rewrite_type="semantic"
        )

    def _rewrite_simple(self, question: str) -> RewriteResult:
        """简单替换（fallback）"""
        rewritten = question.strip()
        if "？" in rewritten:
            rewritten = rewritten.replace("？", "?")
        if "!" in rewritten:
            rewritten = rewritten.replace("!", "。")
        return RewriteResult(
            original_question=question,
            rewritten_question=rewritten,
            rewrite_type="simple"
        )

    def evaluate_retrieval_sufficiency(self, chunks: List[Any], question: str, scores: List[float] = None) -> SufficiencyResult:
        """评估检索结果充分性"""
        if not chunks:
            return SufficiencyResult(
                is_sufficient=False,
                reasoning="未检索到任何相关文档",
                missing_aspects=["相关知识文档"],
                suggestions=["建议补充相关知识文档", "尝试使用不同的关键词检索"]
            )

        valid_scores = [float(score) for score in (scores or []) if score is not None]
        if valid_scores and max(valid_scores) <= 0:
            return SufficiencyResult(
                is_sufficient=False,
                reasoning="重排序结果均无关键词匹配，未检索到可靠资料",
                missing_aspects=["高质量检索结果"],
                suggestions=["优化检索问题描述", "补充相关知识文档"]
            )

        coverage = min(1.0, len(chunks) * 0.3)
        if coverage < 0.5:
            return SufficiencyResult(
                is_sufficient=False,
                reasoning=f"检索结果覆盖度较低（{coverage:.2f}）",
                missing_aspects=["相关文档数量"],
                suggestions=["增加知识库内容", "调整相似度阈值"]
            )

        return SufficiencyResult(
            is_sufficient=True,
            reasoning=f"检索到{len(chunks)}个相关结果，满足数量与重排序分数规则",
            missing_aspects=[],
            suggestions=[]
        )

    def plan_steps(self, state: AgentState, complexity: str = "simple") -> List[str]:
        """
        规划执行步骤
        """
        has_conversation = bool(state.conversation_id)
        is_contextual = complexity == "medium"

        # 附件已经在上传阶段完成解析。总结类直接生成；客服案例类使用
        # Router 同一次分类得到的 retrieval_query 检索企业规则，不再重复改写。
        if state.attachments:
            steps = []
            if state.needs_knowledge_search:
                steps.extend(["knowledge_search", "result_evaluation"])
            steps.append("answer_generation")
            if has_conversation:
                steps.append("memory_write")
            return steps

        steps = []
        if is_contextual and has_conversation:
            steps.append("memory_read")

        # 对于非常短的问题，增加一个澄清步骤
        if len(state.original_input or "") < 4:
            steps.append("clarification")

        # 只有上下文相关问题才需要额外的模型改写。
        if is_contextual:
            steps.append("question_rewrite")

        steps.extend([
            "knowledge_search",
            "result_evaluation",
            "answer_generation"
        ])

        if has_conversation:
            steps.append("memory_write")

        return steps

    def should_terminate(self, state: AgentState) -> Tuple[bool, str]:
        """判断是否应该终止"""
        reason = TerminationCondition.get_termination_reason(state)
        return TerminationCondition.should_terminate(state), reason
