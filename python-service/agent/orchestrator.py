from typing import Dict, Any, List, Optional, Generator, Callable
from agent.state import AgentState, AgentStatus, StepType, TerminationCondition
from agent.planner import Planner
from intent.classifier import IntentType
from agent.executor import Executor
from agent.policies import guardrails
import time
import logging
import uuid
import json

logger = logging.getLogger(__name__)


class Orchestrator:
    """Agent编排器 - 负责协调整个Agent执行流程"""

    def __init__(self):
        self.planner = Planner()
        self.executor = Executor()

    def create_state(self, input_text: str, conversation_id: Optional[str] = None,
                    user_id: Optional[str] = None, goal: Optional[str] = None,
                    run_id: Optional[str] = None, trace_id: Optional[str] = None,
                    attachments: Optional[List[Dict[str, Any]]] = None,
                    attachment_action: Optional[str] = None,
                    needs_knowledge_search: bool = True,
                    retrieval_query: Optional[str] = None) -> AgentState:
        """创建Agent状态"""
        state = AgentState(
            run_id=run_id or str(uuid.uuid4()),
            trace_id=trace_id or str(uuid.uuid4()),
            conversation_id=conversation_id,
            user_id=user_id,
            goal=goal or f"回答用户问题: {input_text[:50]}...",
            original_input=input_text,
            attachments=attachments or [],
            attachment_action=attachment_action,
            needs_knowledge_search=needs_knowledge_search,
            retrieval_query=retrieval_query,
            status=AgentStatus.PENDING
        )
        return state

    def run_stream(self, input_text: str, conversation_id: Optional[str] = None,
                  user_id: Optional[str] = None, context: str = "",
                  goal: Optional[str] = None, run_id: Optional[str] = None,
                  trace_id: Optional[str] = None, complexity: str = "simple",
                  attachments: Optional[List[Dict[str, Any]]] = None,
                  attachment_action: Optional[str] = None,
                  needs_knowledge_search: bool = True,
                  retrieval_query: Optional[str] = None,
                  **kwargs) -> Generator[str, None, None]:
        """流式执行Agent"""
        state = self.create_state(
            input_text, conversation_id, user_id, goal, run_id, trace_id,
            attachments=attachments,
            attachment_action=attachment_action,
            needs_knowledge_search=needs_knowledge_search,
            retrieval_query=retrieval_query,
        )
        state.context = context

        is_valid, error_msg = guardrails.check_input(input_text)
        if not is_valid:
            state.fail(error_msg or "输入验证失败", "INPUT_VALIDATION_ERROR")
            yield json.dumps({
                "type": "error",
                "content": error_msg
            })
            return

        state.start()
        try:
            planned_steps = self.planner.plan_steps(state, complexity=complexity)
            state.planned_steps = planned_steps
            stream_result: Optional[Dict[str, Any]] = None

            for step_name in planned_steps:
                step_type = self._get_step_type(step_name)
                step = state.add_step(step_type, step_name, {"input": input_text})

                yield json.dumps({
                    "type": "step_started",
                    "step_name": step_name,
                    "step_type": step_type.value
                })

                try:
                    # answer_generation 步骤特殊处理：用流式生成，而不是 execute_step 一次性生成
                    if step_type == StepType.ANSWER_GENERATION:
                        question = state.original_input or ""
                        context = state.context or ""
                        attachment_context = self._format_attachment_context(state.attachments)
                        attachment_sources = [
                            {
                                "filename": attachment.get("filename", "未命名附件"),
                                "source_type": "attachment",
                            }
                            for attachment in state.attachments
                            if attachment.get("content") or attachment.get("text")
                        ]

                        # 收集 knowledge_search 步骤的检索结果作为 docs
                        docs = []
                        sources = []
                        retrieval_is_sufficient = True
                        for s in state.steps:
                            if s.step_type == StepType.RESULT_EVALUATION and s.output_data:
                                retrieval_is_sufficient = s.output_data.get("is_sufficient", True)
                                break

                        for s in state.steps:
                            if s.step_type == StepType.KNOWLEDGE_SEARCH and s.output_data:
                                chunks = s.output_data.get("chunks", [])
                                sources = s.output_data.get("sources", [])
                                for chunk in chunks:
                                    doc = type('Doc', (), {
                                        'page_content': chunk.get('content', ''),
                                        'metadata': {'source': '', 'doc_id': '', 'page': ''}
                                    })()
                                    docs.append(doc)
                                break

                        if not retrieval_is_sufficient and not attachment_context and not context.strip():
                            full_answer = "抱歉，未检索到足够可靠的知识库资料，无法基于当前资料回答。请换一种描述，或补充相关文档。"
                            sources = []
                        else:
                            if not retrieval_is_sufficient:
                                docs = []
                                sources = []
                            # 用流式生成：边生成边 yield token
                            full_answer = ""
                            for evt_str in self.executor.llm_service.get_answer_stream(
                                question,
                                docs,
                                context,
                                attachment_context=attachment_context,
                                attachment_action=state.attachment_action or "none",
                            ):
                                try:
                                    evt = json.loads(evt_str)
                                    if evt.get("type") == "token":
                                        full_answer += evt.get("content", "")
                                        yield json.dumps({
                                            "type": "token",
                                            "content": evt.get("content", "")
                                        })
                                    elif evt.get("type") == "end":
                                        full_answer = evt.get("content", full_answer)
                                except Exception:
                                    pass

                        sources = attachment_sources + sources
                        stream_result = {
                            "answer": full_answer,
                            "sources": sources,
                            "has_sources": bool(sources),
                            "task_type": "knowledge_qa"
                        }
                        step.complete(stream_result)

                        yield json.dumps({
                            "type": "step_completed",
                            "step_name": step_name,
                            "output": step.output_data
                        })
                        yield json.dumps({
                            "type": "sources",
                            "sources": sources
                        })
                        continue

                    self.executor.execute_step(state, step)

                    yield json.dumps({
                        "type": "step_completed",
                        "step_name": step_name,
                        "output": step.output_data
                    })

                    if state.status == AgentStatus.WAITING:
                        clarification_output = self._handle_clarification(state)
                        state.complete(clarification_output)
                        yield json.dumps({
                            "type": "clarification",
                            "content": clarification_output
                        })
                        break

                except Exception as e:
                    logger.error(f"[{state.run_id}] Step {step_name} failed: {str(e)}")
                    yield json.dumps({
                        "type": "step_failed",
                        "step_name": step_name,
                        "error": str(e)
                    })
                    state.fail(str(e), "STEP_EXECUTION_ERROR")
                    break

                should_terminate, reason = self.planner.should_terminate(state)
                if should_terminate:
                    break

            if state.status == AgentStatus.RUNNING and stream_result is not None:
                state.complete(stream_result)
                yield json.dumps({
                    "type": "end",
                    "content": stream_result
                })

        except Exception as e:
            logger.error(f"[{state.run_id}] Orchestrator stream run failed: {str(e)}")
            yield json.dumps({
                "type": "error",
                "content": str(e)
            })
            state.fail(str(e), "ORCHESTRATOR_ERROR")

    @staticmethod
    def _format_attachment_context(attachments: List[Dict[str, Any]]) -> str:
        """把临时附件格式化为独立事实材料，不混入对话历史。"""
        parts = []
        for attachment in attachments:
            filename = attachment.get("filename", "未命名附件")
            content = str(attachment.get("content") or attachment.get("text") or "").strip()
            if content:
                parts.append(f"【附件：{filename}】\n{content}")
        return "\n\n".join(parts)[:50000]

    def _get_step_type(self, step_name: str) -> StepType:
        """获取步骤类型"""
        step_mapping = {
            "clarification": StepType.CLARIFICATION,
            "question_rewrite": StepType.QUESTION_REWRITE,
            "knowledge_search": StepType.KNOWLEDGE_SEARCH,
            "result_evaluation": StepType.RESULT_EVALUATION,
            "answer_generation": StepType.ANSWER_GENERATION,
            "memory_read": StepType.MEMORY_READ,
            "memory_write": StepType.MEMORY_WRITE,
            "identity_answer": StepType.ANSWER_GENERATION,
            "admin_operation": StepType.TOOL_CALL
        }
        return step_mapping.get(step_name, StepType.TOOL_CALL)

    def _handle_clarification(self, state: AgentState) -> Dict[str, Any]:
        """处理澄清请求"""
        clarification_step = None
        for step in reversed(state.steps):
            if step.step_type == StepType.CLARIFICATION:
                clarification_step = step
                break

        if clarification_step and clarification_step.output_data.get("needs_clarification"):
            prompt = clarification_step.output_data.get("prompt", "请提供更多信息")
            return {
                "type": "clarification",
                "content": prompt,
                "requires_input": True
            }

        return {
            "type": "clarification",
            "content": "请详细描述您的问题",
            "requires_input": True
        }


