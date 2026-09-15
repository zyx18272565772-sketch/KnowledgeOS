from typing import Dict, Any, Optional, List
from agent.state import AgentState, AgentStep, StepType, StepStatus
from agent.planner import Planner, RewriteResult, SufficiencyResult
from tools.registry import tool_registry
from core.llm import LLMService
from core.vector_store import vector_store
from core.config import config
import time
import logging
import json
import os

logger = logging.getLogger(__name__)


class Executor:
    """步骤执行器 - 负责执行各个步骤"""

    def __init__(self):
        self.planner = Planner()
        self.llm_service = LLMService()
        self.vector_store = vector_store
        self._memory_agent = None

    @property
    def memory_agent(self):
        """延迟加载 MemoryAgent"""
        if self._memory_agent is None:
            from agent.memory_agent import MemoryAgent
            self._memory_agent = MemoryAgent()
        return self._memory_agent

    def execute_step(self, state: AgentState, step: AgentStep) -> Dict[str, Any]:
        """执行单个步骤"""
        step.start()
        logger.info(f"[{state.run_id}] Executing step: {step.step_name} (type: {step.step_type.value})")

        try:
            if step.step_type == StepType.CLARIFICATION:
                return self._execute_clarification(state, step)
            elif step.step_type == StepType.QUESTION_REWRITE:
                return self._execute_question_rewrite(state, step)
            elif step.step_type == StepType.KNOWLEDGE_SEARCH:
                return self._execute_knowledge_search(state, step)
            elif step.step_type == StepType.RESULT_EVALUATION:
                return self._execute_result_evaluation(state, step)
            elif step.step_type == StepType.MEMORY_READ:
                return self._execute_memory_read(state, step)
            elif step.step_type == StepType.MEMORY_WRITE:
                return self._execute_memory_write(state, step)
            elif step.step_type == StepType.TOOL_CALL:
                return self._execute_tool_call(state, step)
            else:
                raise ValueError(f"Unknown step type: {step.step_type}")
        except Exception as e:
            logger.error(f"[{state.run_id}] Step {step.step_name} failed: {str(e)}")
            step.fail(str(e))
            raise

    def _execute_clarification(self, state: AgentState, step: AgentStep) -> Dict[str, Any]:
        """执行澄清判断"""
        intent = self.planner.recognize_intent(state)
        needs_clarification, prompt = self.planner.check_clarification_needed(state, intent)

        step.complete({
            "needs_clarification": needs_clarification,
            "prompt": prompt
        })

        if needs_clarification:
            state.wait()

        return step.output_data

    def _execute_question_rewrite(self, state: AgentState, step: AgentStep) -> Dict[str, Any]:
        """执行问题改写"""
        question = state.original_input or ""
        rewrite_result = self.planner.rewrite_question(question, conversation_context=state.context or "")
        state.add_intermediate_conclusion(
            step_id=step.step_id,
            conclusion_type="rewritten_question",
            content=rewrite_result.rewritten_question
        )

        step.complete({
            "original_question": rewrite_result.original_question,
            "rewritten_question": rewrite_result.rewritten_question,
            "rewrite_type": rewrite_result.rewrite_type
        })

        return step.output_data

    def _execute_knowledge_search(self, state: AgentState, step: AgentStep) -> Dict[str, Any]:
        """执行知识检索"""
        query = state.original_input or ""

        rewritten_question = None
        for conclusion in state.intermediate_conclusions:
            if conclusion.conclusion_type == "rewritten_question":
                rewritten_question = conclusion.content
                break

        search_query = state.retrieval_query or rewritten_question or query

        if tool_registry.has_tool("knowledge_search"):
            try:
                result = tool_registry.invoke_tool(
                    "knowledge_search",
                    {"query": search_query, "top_k": config.KNOWLEDGE_TOP_K}
                )
                chunks = result.get("chunks", [])
                scores = result.get("scores", [])
            except Exception as e:
                logger.warning(f"[{state.run_id}] knowledge_search tool failed: {e}")
                docs = self.vector_store.search(search_query, k=config.KNOWLEDGE_TOP_K)
                chunks = [
                    {
                        "content": doc.page_content,
                        "metadata": doc.metadata,
                        "score": doc.metadata.get("rerank_score")
                    }
                    for doc in docs
                ]
                scores = [chunk["score"] for chunk in chunks]
        else:
            docs = self.vector_store.search(search_query, k=config.KNOWLEDGE_TOP_K)
            chunks = [
                {
                    "content": doc.page_content,
                    "metadata": doc.metadata,
                    "score": doc.metadata.get("rerank_score")
                }
                for doc in docs
            ]
            scores = [chunk["score"] for chunk in chunks]

        sources = []
        for chunk in chunks:
            metadata = chunk.get("metadata", {})
            source_info = {
                "source": metadata.get('source', ''),
                "doc_id": metadata.get('doc_id', ''),
                "page": metadata.get('page', '')
            }
            if source_info["source"]:
                source_info["doc_name"] = os.path.basename(source_info["source"])
            sources.append(source_info)

        step.complete({
            "chunks": chunks,
            "scores": scores,
            "sources": sources,
            "count": len(chunks)
        })

        return step.output_data

    def _execute_result_evaluation(self, state: AgentState, step: AgentStep) -> Dict[str, Any]:
        """执行结果充分性判断"""
        chunks = None
        scores = None
        for s in state.steps:
            if s.step_type == StepType.KNOWLEDGE_SEARCH and s.output_data:
                chunks = s.output_data.get("chunks", [])
                scores = s.output_data.get("scores", [])
                break

        if chunks is None:
            step.complete({
                "is_sufficient": False,
                "reasoning": "未找到检索结果"
            })
            return step.output_data

        sufficiency = self.planner.evaluate_retrieval_sufficiency(
            [type('obj', (object,), {'page_content': c.get('content', '')}) for c in chunks],
            state.original_input or "",
            scores
        )

        state.add_intermediate_conclusion(
            step_id=step.step_id,
            conclusion_type="sufficiency",
            content={
                "is_sufficient": sufficiency.is_sufficient,
                "reasoning": sufficiency.reasoning
            }
        )

        step.complete({
            "is_sufficient": sufficiency.is_sufficient,
            "reasoning": sufficiency.reasoning,
            "missing_aspects": sufficiency.missing_aspects,
            "suggestions": sufficiency.suggestions
        })

        return step.output_data

    def _execute_memory_write(self, state: AgentState, step: AgentStep) -> Dict[str, Any]:
        """执行记忆写入（使用 Memory Agent）"""
        answer = None
        for s in reversed(state.steps):
            if s.step_type == StepType.ANSWER_GENERATION and s.output_data:
                answer = s.output_data.get("answer", "")
                break

        # 使用 Memory Agent 保存记忆
        self.memory_agent.save_memory(state, state.original_input, answer)

        step.complete({
            "success": True,
            "message": "记忆写入完成"
        })

        return step.output_data

    def _execute_memory_read(self, state: AgentState, step: AgentStep) -> Dict[str, Any]:
        """执行记忆读取（使用 Memory Agent）"""
        context = self.memory_agent.load_memory(state)

        step.complete({
            "context": context,
            "has_history": bool(context)
        })

        # 将记忆上下文保存到 state，供后续步骤使用
        if context:
            state.context = f"{state.context}\n\n{context}" if state.context else context

        return step.output_data

    def _execute_tool_call(self, state: AgentState, step: AgentStep) -> Dict[str, Any]:
        """执行通用工具调用"""
        tool_name = step.input_data.get("tool_name")
        parameters = step.input_data.get("parameters", {})

        if not tool_registry.has_tool(tool_name):
            raise ValueError(f"Tool not found: {tool_name}")

        result = tool_registry.invoke_tool(tool_name, parameters)

        step.complete({
            "result": result,
            "tool_name": tool_name
        })

        return step.output_data
