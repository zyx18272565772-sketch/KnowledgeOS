"""测试临时附件问答链路。"""

import json
from unittest.mock import Mock, patch

from agent.executor import Executor
from agent.orchestrator import Orchestrator
from agent.planner import Planner
from agent.state import AgentState, StepType
from intent.classifier import IntentClassifier


def make_attachment(content: str = "用户收到商品时发现破损，要求退款。",
                    cached: bool = False) -> dict:
    attachment = {
        "filename": "售后聊天.png",
        "file_type": "image",
        "content": content,
        "chunks_count": 1,
        "char_count": len(content),
    }
    if cached:
        attachment["_from_cache"] = True
    return attachment


def test_attachment_summary_fallback_skips_knowledge_search():
    classifier = IntentClassifier()
    classifier._llm_service = False

    result = classifier.classify("这张截图有什么内容？", [make_attachment()])

    assert result.attachment_action == "summary"
    assert result.needs_knowledge_search is False
    assert result.retrieval_query is None


def test_attachment_case_fallback_builds_retrieval_query():
    classifier = IntentClassifier()
    classifier._llm_service = False

    result = classifier.classify("这个问题怎么回答？", [make_attachment()])

    assert result.attachment_action == "case_assist"
    assert result.needs_knowledge_search is True
    assert "商品" in result.retrieval_query
    assert "破损" in result.retrieval_query


def test_cached_attachment_follow_up_can_reuse_attachment():
    classifier = IntentClassifier()
    classifier._llm_service = False

    result = classifier.classify("这张截图应该怎么回复？", [make_attachment(cached=True)])

    assert result.attachment_action == "case_assist"
    assert result.needs_knowledge_search is True
    assert "商品" in result.retrieval_query


def test_cached_attachment_does_not_hijack_unrelated_question():
    classifier = IntentClassifier()
    classifier._llm_service = False

    result = classifier.classify("公司的报销规则是什么？", [make_attachment(cached=True)])

    assert result.attachment_action is None
    assert result.needs_knowledge_search is True
    assert result.retrieval_query is None


def test_planner_summary_uses_attachment_without_faiss():
    planner = Planner()
    state = AgentState(
        original_input="总结附件",
        attachments=[make_attachment()],
        attachment_action="summary",
        needs_knowledge_search=False,
    )

    steps = planner.plan_steps(state)

    assert steps == ["answer_generation"]


def test_planner_case_searches_without_second_query_rewrite():
    planner = Planner()
    state = AgentState(
        original_input="应该怎么回复？",
        conversation_id="user_x0010",
        attachments=[make_attachment()],
        attachment_action="case_assist",
        needs_knowledge_search=True,
        retrieval_query="商品破损退款处理规则",
    )

    steps = planner.plan_steps(state)

    assert steps == [
        "knowledge_search",
        "result_evaluation",
        "answer_generation",
        "memory_write",
    ]
    assert "question_rewrite" not in steps


def test_orchestrator_passes_attachment_as_separate_context():
    orchestrator = Orchestrator()
    orchestrator.executor.llm_service.get_answer_stream = Mock(return_value=iter([
        json.dumps({"type": "token", "content": "附件摘要"}),
        json.dumps({"type": "end", "content": "附件摘要"}),
    ]))

    events = list(orchestrator.run_stream(
        input_text="截图中有什么内容？",
        attachments=[make_attachment()],
        attachment_action="summary",
        needs_knowledge_search=False,
    ))

    call = orchestrator.executor.llm_service.get_answer_stream.call_args
    assert call.args[1] == []
    assert "用户收到商品时发现破损" in call.kwargs["attachment_context"]
    assert call.kwargs["attachment_action"] == "summary"
    assert any(json.loads(event).get("type") == "end" for event in events)


@patch("agent.executor.tool_registry")
def test_executor_prefers_router_retrieval_query(mock_registry):
    mock_registry.has_tool.return_value = True
    mock_registry.invoke_tool.return_value = {
        "chunks": [],
        "scores": [],
    }
    state = AgentState(
        original_input="这个问题怎么回答？",
        retrieval_query="商品破损退款处理规则",
    )
    step = state.add_step(StepType.KNOWLEDGE_SEARCH, "knowledge_search")

    Executor()._execute_knowledge_search(state, step)

    _, parameters = mock_registry.invoke_tool.call_args.args
    assert parameters["query"] == "商品破损退款处理规则"
