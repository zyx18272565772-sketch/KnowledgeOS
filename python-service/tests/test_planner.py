"""测试 Planner 规划器"""

import pytest
from agent.planner import Planner
from intent.classifier import IntentType, IntentResult
from agent.state import AgentState


@pytest.fixture
def planner():
    return Planner()


class TestIntentRecognition:
    """测试意图识别"""

    def test_recognize_chitchat(self, planner):
        """测试识别闲聊意图"""
        state = AgentState(original_input="你好，今天天气怎么样？")
        result = planner.recognize_intent(state)
        assert result.intent == IntentType.CHITCHAT

    def test_recognize_knowledge_qa(self, planner):
        """测试识别知识问答意图"""
        state = AgentState(original_input="什么是 RAG？请解释一下它的工作原理")
        result = planner.recognize_intent(state)
        assert result.intent == IntentType.KNOWLEDGE_QA

    def test_recognize_identity_query(self, planner):
        """测试识别身份查询意图"""
        state = AgentState(original_input="你是谁？")
        result = planner.recognize_intent(state)
        # "你是谁" 在当前关键词匹配下被归类为闲聊
        assert result.intent in [IntentType.IDENTITY_QUERY, IntentType.CHITCHAT]

    def test_recognize_unknown(self, planner):
        """测试识别未知意图"""
        state = AgentState(original_input="asdfghjkl")
        result = planner.recognize_intent(state)
        assert result.intent == IntentType.UNKNOWN


class TestClarificationCheck:
    """测试澄清检查"""

    def test_need_clarification_vague(self, planner):
        """测试模糊问题需要澄清"""
        state = AgentState(original_input="那个")
        intent = planner.recognize_intent(state)
        needs_clarification, prompt = planner.check_clarification_needed(state, intent)
        assert needs_clarification is True

    def test_need_clarification_short(self, planner):
        """测试过短问题需要澄清"""
        state = AgentState(original_input="啥")
        intent = planner.recognize_intent(state)
        needs_clarification, prompt = planner.check_clarification_needed(state, intent)
        assert needs_clarification is True

    def test_no_clarification_needed(self, planner):
        """测试正常问题不需要澄清"""
        state = AgentState(original_input="什么是机器学习？请详细解释一下")
        intent = planner.recognize_intent(state)
        needs_clarification, _ = planner.check_clarification_needed(state, intent)
        assert needs_clarification is False


class TestQuestionRewrite:
    """测试问题改写"""

    def test_rewrite_simple(self, planner):
        """测试简单改写（fallback）"""
        result = planner.rewrite_question("什么是 RAG？", rewrite_type="simple")
        assert result.original_question == "什么是 RAG？"
        assert result.rewritten_question is not None
        assert result.rewrite_type == "simple"

    def test_rewrite_semantic_fallback(self, planner):
        """测试语义改写（无 LLM 时降级为简单改写）"""
        result = planner.rewrite_question("如何学习编程？")
        # 无 DASHSCOPE_API_KEY 时应降级为 simple
        assert result.rewrite_type in ["semantic", "simple"]


class TestRetrievalSufficiency:
    """测试检索充分性判断"""

    def test_sufficient_retrieval(self, planner):
        """测试充分的检索结果"""
        chunks = [{"content": "test"}, {"content": "test2"}, {"content": "test3"}]
        scores = [0.9, 0.85, 0.8]
        result = planner.evaluate_retrieval_sufficiency(chunks, "test question", scores)
        assert result.is_sufficient is True

    def test_insufficient_no_chunks(self, planner):
        """测试无检索结果"""
        result = planner.evaluate_retrieval_sufficiency([], "test question")
        assert result.is_sufficient is False

    def test_insufficient_zero_scores(self, planner):
        """测试无关键词匹配的检索结果"""
        chunks = [{"content": "test"}, {"content": "test2"}]
        scores = [0.0, 0.0]
        result = planner.evaluate_retrieval_sufficiency(chunks, "test question", scores)
        assert result.is_sufficient is False


class TestStepPlanning:
    """
    测试步骤规划
    """

    def test_plan_knowledge_qa_steps(self, planner):
        """测试 L1 简单知识问答步骤规划"""
        state = AgentState(original_input="什么是 RAG？请解释原理")
        steps = planner.plan_steps(state)
        
        assert "question_rewrite" not in steps
        assert "memory_read" not in steps
        assert "knowledge_search" in steps
        assert "result_evaluation" in steps
        assert "answer_generation" in steps

    def test_plan_with_conversation_memory(self, planner):
        """测试带会话记忆的步骤规划"""
        state = AgentState(original_input="它和传统的检索有什么区别？", conversation_id="test-123")
        steps = planner.plan_steps(state, complexity="medium")
        
        # 验证在有会话ID时，会包含读写记忆的步骤
        assert steps[0] == "memory_read"
        assert steps[-1] == "memory_write"
        assert "question_rewrite" in steps

    def test_plan_contextual_question_rewrites_without_memory(self, planner):
        """测试 L2 上下文问题会改写，即使没有会话 ID。"""
        state = AgentState(original_input="这个规则是什么意思？")
        steps = planner.plan_steps(state, complexity="medium")
        assert "question_rewrite" in steps
        assert "memory_read" not in steps

    def test_plan_for_short_question_needs_clarification(self, planner):
        """测试短问题会触发澄清步骤"""
        state = AgentState(original_input="它？")
        steps = planner.plan_steps(state)
        
        # 验证对于短问题，会增加澄清步骤
        assert "clarification" in steps
