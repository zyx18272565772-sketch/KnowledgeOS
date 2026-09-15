"""知识巡检：FAISS 语义重复候选与 LLM 降级测试。"""

from unittest.mock import patch

import faiss
import numpy as np
from langchain_core.documents import Document

from core.vector_store import VectorStoreManager
from workflows.inspection_agent import InspectionAgent


class FakeDocstore:
    def __init__(self, documents):
        self._dict = documents

    def search(self, document_id):
        return self._dict.get(document_id)


class FakeLangChainFaiss:
    def __init__(self, vectors, documents):
        dimension = vectors.shape[1] if len(vectors) else 2
        self.index = faiss.IndexFlatL2(dimension)
        if len(vectors):
            self.index.add(np.ascontiguousarray(vectors, dtype=np.float32))

        ids = [f"vector-{index}" for index in range(len(documents))]
        self.index_to_docstore_id = {
            index: document_id for index, document_id in enumerate(ids)
        }
        self.docstore = FakeDocstore(dict(zip(ids, documents)))


def make_document(doc_id, chunk_index, content, source=None):
    return Document(
        page_content=content,
        metadata={
            "doc_id": doc_id,
            "source": source or f"文档{doc_id}.txt",
            "chunk_index": chunk_index,
        },
    )


def make_manager(vectors, documents):
    """绕过 Embedding 初始化，直接挂载已有的内存 FAISS 索引。"""
    manager = VectorStoreManager.__new__(VectorStoreManager)
    manager._initialized = True
    manager.use_milvus = False
    manager.vector_store = FakeLangChainFaiss(
        np.asarray(vectors, dtype=np.float32),
        documents,
    )
    return manager


def make_candidate():
    return {
        "left": {
            "vector_id": "a",
            "doc_id": 1,
            "source": "售后规则.txt",
            "chunk_index": 0,
            "content": "签收后七天内支持无理由退货。",
            "preview": "签收后七天内支持无理由退货。",
        },
        "right": {
            "vector_id": "b",
            "doc_id": 2,
            "source": "客服FAQ.txt",
            "chunk_index": 3,
            "content": "收到商品后的七日内可以无理由退货。",
            "preview": "收到商品后的七日内可以无理由退货。",
        },
        "similarity": 0.96,
    }


def test_self_match_is_not_returned_and_ab_ba_is_one_pair():
    manager = make_manager(
        [[1.0, 0.0], [0.99, 0.01]],
        [
            make_document(1, 0, "规则A"),
            make_document(2, 0, "规则A的同义表达"),
        ],
    )

    pairs = manager.find_semantic_duplicate_pairs(
        top_k=1,
        cosine_threshold=0.90,
        max_candidates=20,
    )

    assert len(pairs) == 1
    assert pairs[0]["left"]["vector_id"] != pairs[0]["right"]["vector_id"]
    assert {pairs[0]["left"]["doc_id"], pairs[0]["right"]["doc_id"]} == {1, 2}


def test_chunks_from_same_document_are_excluded():
    manager = make_manager(
        [[1.0, 0.0], [0.99, 0.01]],
        [
            make_document(1, 0, "同一文档片段A"),
            make_document(1, 1, "同一文档片段B"),
        ],
    )

    assert manager.find_semantic_duplicate_pairs(
        top_k=1,
        cosine_threshold=0.90,
        max_candidates=20,
    ) == []


def test_similarity_below_threshold_is_excluded():
    manager = make_manager(
        [[1.0, 0.0], [0.0, 1.0]],
        [
            make_document(1, 0, "规则A"),
            make_document(2, 0, "完全不同的规则B"),
        ],
    )

    assert manager.find_semantic_duplicate_pairs(
        top_k=1,
        cosine_threshold=0.90,
        max_candidates=20,
    ) == []


def test_high_similarity_cross_document_chunks_become_candidate():
    manager = make_manager(
        [[3.0, 0.0], [6.0, 0.1]],
        [
            make_document(10, 2, "七天无理由退货", "规则.txt"),
            make_document(20, 5, "七日内可以退货", "FAQ.txt"),
        ],
    )

    pairs = manager.find_semantic_duplicate_pairs(
        top_k=1,
        cosine_threshold=0.99,
        max_candidates=20,
    )

    assert len(pairs) == 1
    assert pairs[0]["similarity"] >= 0.99
    assert pairs[0]["left"]["chunk_index"] == 2
    assert pairs[0]["right"]["chunk_index"] == 5


def test_empty_faiss_returns_no_candidates():
    manager = make_manager([], [])

    assert manager.find_semantic_duplicate_pairs() == []


def test_existing_knowledge_search_path_is_unchanged():
    expected = make_document(1, 0, "正常知识问答检索结果")

    class SearchableVectorStore:
        @staticmethod
        def similarity_search_with_score(query, k):
            assert query == "报销规则"
            assert k == 1
            return [(expected, 0.12)]

    manager = VectorStoreManager.__new__(VectorStoreManager)
    manager._initialized = True
    manager.use_milvus = False
    manager.vector_store = SearchableVectorStore()
    manager.reranker = None

    result = manager.search("报销规则", k=1, use_rerank=False)

    assert result == [expected]


@patch("workflows.inspection_agent.llm_service.chat", side_effect=TimeoutError("timeout"))
@patch("workflows.inspection_agent.vector_store.find_semantic_duplicate_pairs")
@patch("workflows.inspection_agent.mysql_client.fetch_all", return_value=[])
def test_llm_failure_degrades_to_suspected_duplicate(
    _mock_fetch_all,
    mock_find_pairs,
    _mock_chat,
):
    mock_find_pairs.return_value = [make_candidate()]

    with patch("workflows.inspection_agent.config.DUPLICATE_LLM_VERIFY", True):
        result = InspectionAgent()._check_duplicate_docs()

    assert result.get("error") is not True
    assert result["data"]["semantic_count"] == 1
    assert result["data"]["semantic_confirmed_count"] == 0
    assert result["data"]["semantic_suspected_count"] == 1
    assert result["data"]["semantic_duplicates"][0]["status"] == "suspected"
    assert "疑似重复" in result["answer"]


@patch(
    "workflows.inspection_agent.llm_service.chat",
    return_value='```json\n{"is_duplicate": false, "reason": "期限不同"}\n```',
)
@patch("workflows.inspection_agent.vector_store.find_semantic_duplicate_pairs")
@patch("workflows.inspection_agent.mysql_client.fetch_all", return_value=[])
def test_llm_can_reject_vector_candidate(_mock_fetch_all, mock_find_pairs, _mock_chat):
    mock_find_pairs.return_value = [make_candidate()]

    with patch("workflows.inspection_agent.config.DUPLICATE_LLM_VERIFY", True):
        result = InspectionAgent()._check_duplicate_docs()

    assert result["data"]["semantic_count"] == 0
    assert result["data"]["semantic_rejected_count"] == 1


@patch("workflows.inspection_agent.llm_service.chat")
@patch("workflows.inspection_agent.vector_store.find_semantic_duplicate_pairs")
@patch("workflows.inspection_agent.mysql_client.fetch_all")
def test_title_duplicate_is_not_counted_again_as_semantic(
    mock_fetch_all,
    mock_find_pairs,
    mock_chat,
):
    mock_fetch_all.return_value = [{
        "doc1_id": 1,
        "doc1_title": "退款规范",
        "doc2_id": 2,
        "doc2_title": "退款规范",
    }]
    mock_find_pairs.return_value = [make_candidate()]

    result = InspectionAgent()._check_duplicate_docs()

    assert result["data"]["title_count"] == 1
    assert result["data"]["semantic_count"] == 0
    assert result["data"]["count"] == 1
    mock_chat.assert_not_called()


@patch("workflows.inspection_agent.vector_store.find_semantic_duplicate_pairs", return_value=[])
@patch("workflows.inspection_agent.mysql_client.fetch_all")
def test_original_exact_title_duplicate_detection_is_preserved(
    mock_fetch_all,
    _mock_find_pairs,
):
    mock_fetch_all.return_value = [{
        "doc1_id": 11,
        "doc1_title": "退款规范",
        "doc2_id": 12,
        "doc2_title": "退款规范",
    }]

    result = InspectionAgent()._check_duplicate_docs()

    assert result["data"]["title_count"] == 1
    assert result["data"]["semantic_count"] == 0
    assert result["data"]["count"] == 1
    assert "《退款规范》" in result["answer"]


def test_full_inspection_uses_upgraded_duplicate_count():
    agent = InspectionAgent()
    duplicate_result = {"data": {"count": 3}}
    low_quality_result = {"data": {"count": 2}}
    stale_result = {"data": {"count": 1}}

    with patch.object(agent, "_check_duplicate_docs", return_value=duplicate_result), \
         patch.object(agent, "_check_low_quality_chunks", return_value=low_quality_result), \
         patch.object(agent, "_check_stale_knowledge", return_value=stale_result):
        result = agent._run_full_inspection()

    assert result["data"]["duplicate_count"] == 3
    assert result["data"]["total_issues"] == 6
