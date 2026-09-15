from typing import Dict, Any, Optional, List
from core.llm import LLMService
from core.vector_store import vector_store
from core.config import config
from tools.registry import tool_registry
from agent.planner import Planner
import logging
import json

logger = logging.getLogger(__name__)


class ReasoningAgent:
    """Reasoning Agent - 处理复杂问题的分步推理（Chain-of-Thought）"""

    def __init__(self):
        self.llm_service = LLMService()
        self.vector_store = vector_store
        self.planner = Planner()

    def reason_stream(self, question: str, context: str = "",
                      conversation_id: str = None):
        """流式推理：分解→逐个推理（有进度）→流式汇总"""
        logger.info(f"[ReasoningAgent] Streaming reasoning: {question[:50]}...")

        try:
            # Step 1：问题分解
            yield json.dumps({"type": "token", "content": "🔍 正在拆解问题…\n\n"})
            sub_questions = self._decompose_question(question, context)
            yield json.dumps({"type": "token", "content": f"已拆解为 {len(sub_questions)} 个子问题：\n"})
            for i, sq in enumerate(sub_questions, 1):
                yield json.dumps({"type": "token", "content": f"  {i}. {sq}\n"})

            # Step 2：逐个子问题推理
            reasoning_steps = []
            for i, sub_q in enumerate(sub_questions, 1):
                yield json.dumps({"type": "token", "content": f"\n⏳ 正在推理子问题 {i}/{len(sub_questions)}…\n"})
                docs = self.vector_store.search(sub_q, k=config.REASONING_TOP_K)
                scores = [
                    doc.metadata.get("rerank_score")
                    for doc in docs
                    if getattr(doc, "metadata", None)
                ]
                sufficiency = self.planner.evaluate_retrieval_sufficiency(docs, sub_q, scores)

                if sufficiency.is_sufficient:
                    sub_answer = self._reason_sub_question(sub_q, docs, context)
                    step_sources = [
                        {"doc_id": getattr(d, 'metadata', {}).get("doc_id"),
                         "doc": getattr(d, 'metadata', {}).get("source", "未知文档")}
                        for d in docs
                    ]
                else:
                    sub_answer = f"该子问题未检索到足够可靠的知识库资料（{sufficiency.reasoning}），不作推测性回答。"
                    step_sources = []

                reasoning_steps.append({
                    "sub_question": sub_q,
                    "sources": step_sources,
                    "reasoning": sub_answer,
                    "is_sufficient": sufficiency.is_sufficient,
                })

            # Step 3：流式汇总生成
            yield json.dumps({"type": "token", "content": f"\n✍️ 正在汇总生成答案…\n\n---\n\n"})

            all_sources = []
            seen_ids = set()
            for step in reasoning_steps:
                for src in step["sources"]:
                    if src["doc_id"] not in seen_ids:
                        seen_ids.add(src["doc_id"])
                        all_sources.append(src)

            reliable_steps = [step for step in reasoning_steps if step["is_sufficient"]]
            if not reliable_steps:
                final_answer = [json.dumps({
                    "type": "token",
                    "content": "抱歉，知识库中未检索到足够可靠的资料，无法基于当前资料回答该问题。"
                })]
            else:
                final_answer = self._synthesize_answer_stream(question, reasoning_steps)
            full_text = ""
            for chunk in final_answer:
                yield chunk
                try:
                    data = json.loads(chunk)
                    if data.get("type") == "token":
                        full_text += data.get("content", "")
                except:
                    pass

            yield json.dumps({"type": "end", "content": full_text})
            yield json.dumps({"type": "sources", "sources": all_sources, "task_type": "reasoning"})

            # 保存记忆
            if conversation_id and full_text and tool_registry.has_tool("conversation_memory_write"):
                try:
                    tool_registry.invoke_tool("conversation_memory_write",
                        {"conversation_id": conversation_id, "role": "user", "content": question})
                    tool_registry.invoke_tool("conversation_memory_write",
                        {"conversation_id": conversation_id, "role": "assistant", "content": full_text})
                except:
                    pass

        except Exception as e:
            logger.error(f"[ReasoningAgent] Stream reasoning failed: {e}", exc_info=True)
            yield json.dumps({"type": "error", "content": "处理复杂问题时遇到错误，请稍后再试。"})

    def _synthesize_answer_stream(self, original_question: str, reasoning_steps: list):
        """流式汇总推理结果"""
        if not self.llm_service.llm:
            text = "\n\n".join([s["reasoning"] for s in reasoning_steps])
            for char in text:
                yield json.dumps({"type": "token", "content": char})
            return

        steps_text = "\n".join([
            f"Q: {s['sub_question']}\nA: {s['reasoning']}"
            for s in reasoning_steps
        ])

        from langchain_core.prompts import PromptTemplate
        from langchain_core.output_parsers import StrOutputParser

        prompt = PromptTemplate.from_template(
            """基于以下分步推理结果，回答用户的原始问题。

分步推理：
{steps_text}

原始问题：{original_question}

请只基于有资料支撑的分步结果作答。若某个子问题的结果明确表示资料不足，必须说明该部分无法根据知识库确认，不得自行补充或猜测。
请给出完整、准确、结构化的回答。"""
        )

        chain = prompt | self.llm_service.llm | StrOutputParser()
        for chunk in chain.stream({
            "steps_text": steps_text,
            "original_question": original_question
        }):
            yield json.dumps({"type": "token", "content": chunk})

    def _decompose_question(self, question: str, context: str = "") -> List[str]:
        """将复杂问题分解为子问题"""
        if not self.llm_service.llm:
            return [question]

        from langchain_core.prompts import PromptTemplate
        from langchain_core.output_parsers import StrOutputParser

        prompt = PromptTemplate.from_template(
            """请结合附件或对话上下文，将以下复杂问题分解为 2-4 个简单的子问题，便于逐个检索和推理。

附件或对话上下文：
{context}

问题：{question}

请以 JSON 数组格式输出子问题列表，例如：["子问题1", "子问题2"]
只输出 JSON 数组，不要其他内容。"""
        )

        try:
            chain = prompt | self.llm_service.llm | StrOutputParser()
            response = chain.invoke({
                "question": question,
                "context": (context or "无")[:6000],
            }).strip()

            # 处理可能的 markdown 代码块
            if response.startswith("```"):
                response = response.split("\n", 1)[1] if "\n" in response else response[3:]
                response = response.rsplit("```", 1)[0]

            sub_questions = json.loads(response.strip())
            if isinstance(sub_questions, list) and len(sub_questions) > 0:
                return sub_questions[:4]
        except Exception as e:
            logger.warning(f"[ReasoningAgent] Failed to decompose question: {e}")

        return [question]

    def _reason_sub_question(self, question: str, docs: list, context: str) -> str:
        """对单个子问题进行推理"""
        if not self.llm_service.llm:
            return "无法推理（LLM 不可用）"

        doc_text = "\n".join([
            getattr(doc, 'page_content', str(doc)) if hasattr(doc, 'page_content')
            else doc.get('content', str(doc)) if isinstance(doc, dict) else str(doc)
            for doc in docs
        ]) if docs else "（无相关文档）"

        from langchain_core.prompts import PromptTemplate
        from langchain_core.output_parsers import StrOutputParser

        prompt = PromptTemplate.from_template(
            """基于以下参考资料，回答子问题。

参考资料：
{doc_text}

对话上下文：{context}

子问题：{question}

请给出简洁准确的回答。"""
        )

        chain = prompt | self.llm_service.llm | StrOutputParser()
        return chain.invoke({
            "doc_text": doc_text,
            "context": context or "无",
            "question": question
        }).strip()

