from typing import Dict, Any, List, Optional, Generator
from agent.orchestrator import Orchestrator
import logging
import json

logger = logging.getLogger(__name__)


class KnowledgeQAAgent:
    """知识问答Agent - 三级链路：L1简化/L2标准/L3推理"""

    def __init__(self):
        self.orchestrator = Orchestrator()

    def ask_stream(self, question: str, conversation_id: Optional[str] = None,
                   user_id: Optional[str] = None, context: str = "",
                   attachments: Optional[List[Dict[str, Any]]] = None,
                   attachment_action: Optional[str] = None,
                   needs_knowledge_search: bool = True,
                   retrieval_query: Optional[str] = None,
                   **kwargs) -> Generator[str, None, None]:
        """
        流式知识问答 - 走编排器完整流程
        """
        logger.info(f"[KnowledgeQAAgent] Stream QA via Orchestrator: {question[:50]}...")

        try:
            kwargs.pop("goal", None)
            complexity = kwargs.pop("complexity", "simple")
            for event in self.orchestrator.run_stream(
                input_text=question,
                conversation_id=conversation_id,
                user_id=user_id,
                context=context,
                attachments=attachments or [],
                attachment_action=attachment_action,
                needs_knowledge_search=needs_knowledge_search,
                retrieval_query=retrieval_query,
                goal=f"回答知识问题: {question[:50]}...",
                complexity=complexity,
                **kwargs
            ):
                yield event
        except Exception as e:
            logger.error(f"[KnowledgeQAAgent] Stream QA via Orchestrator failed: {e}", exc_info=True)
            yield json.dumps({
                "type": "error",
                "content": f"处理问题时遇到错误: {str(e)}"
            })

