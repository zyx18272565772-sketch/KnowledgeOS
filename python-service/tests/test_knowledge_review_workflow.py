"""企业知识上传、巡检、审核与发布闭环测试。"""

import asyncio
import io
import threading
from unittest.mock import Mock, patch

import numpy as np
from fastapi import BackgroundTasks, UploadFile
from langchain_core.documents import Document
from langchain_core.embeddings import Embeddings

from api import document_routes
from core.vector_store import VectorStoreManager
from workflows.inspection_agent import InspectionAgent


class FakeEmbeddings(Embeddings):
    def embed_documents(self, texts):
        return [[1.0, float(index) / 100] for index, _ in enumerate(texts)]

    def embed_query(self, text):
        return [1.0, 0.0]


def make_manager(tmp_path):
    manager = VectorStoreManager.__new__(VectorStoreManager)
    manager._initialized = True
    manager._initialization_lock = threading.Lock()
    manager._mutation_lock = threading.Lock()
    manager.use_milvus = False
    manager.vector_store = None
    manager.embeddings = FakeEmbeddings()
    manager.persist_directory = str(tmp_path / "faiss")
    return manager


def test_pending_embeddings_do_not_enter_online_faiss(tmp_path):
    manager = make_manager(tmp_path)
    documents = [Document(page_content="七天无理由退货", metadata={"doc_id": 9})]
    with patch("core.vector_store.config.PENDING_VECTOR_DIR", str(tmp_path / "pending")):
        vectors = manager.create_pending_embeddings(9, documents)
        assert vectors.shape == (1, 2)
        assert manager.vector_store is None
        assert manager.contains_document(9) is False


def test_approve_reuses_vectors_and_makes_document_searchable(tmp_path):
    manager = make_manager(tmp_path)
    document = Document(
        page_content="商品签收后七天内支持无理由退货",
        metadata={"doc_id": 11, "source": "售后规则.txt", "chunk_index": 0},
    )
    manager.publish_precomputed_documents([document], np.array([[1.0, 0.0]], dtype=np.float32))

    assert manager.contains_document(11) is True
    results = manager.search("退货", k=1, use_rerank=False)
    assert results[0].metadata["doc_id"] == 11


def test_active_match_filters_non_active_documents(tmp_path):
    manager = make_manager(tmp_path)
    manager.publish_precomputed_documents(
        [
            Document(page_content="规则A", metadata={"doc_id": 1, "source": "A", "chunk_index": 0}),
            Document(page_content="规则B", metadata={"doc_id": 2, "source": "B", "chunk_index": 0}),
        ],
        np.array([[1.0, 0.0], [1.0, 0.01]], dtype=np.float32),
    )
    matches = manager.find_active_matches_by_vectors(
        np.array([[1.0, 0.0]], dtype=np.float32),
        active_doc_ids=[1],
        top_k=2,
        cosine_threshold=0.9,
    )
    assert matches
    assert {item["old"]["doc_id"] for item in matches} == {1}


def test_empty_active_faiss_returns_no_candidates(tmp_path):
    manager = make_manager(tmp_path)
    assert manager.find_active_matches_by_vectors(
        np.array([[1.0, 0.0]], dtype=np.float32),
        active_doc_ids=[],
    ) == []


def test_pending_inspection_counts_rules_not_chunks():
    agent = InspectionAgent()
    document = {"id": 7, "status": "PENDING_REVIEW", "title": "新FAQ", "filename": "faq.txt"}
    chunks = [
        {"chunk_index": 0, "chunk_text": "商品签收后七天内支持无理由退货" * 5},
        {"chunk_index": 1, "chunk_text": "退款将在七个工作日内到账" * 6},
    ]
    candidates = [
        {"new_chunk_index": 0, "similarity": 0.96, "old": {"doc_id": 1, "source": "旧规则", "chunk_index": 0, "content": "七日退货", "preview": "七日退货"}},
        {"new_chunk_index": 1, "similarity": 0.93, "old": {"doc_id": 2, "source": "退款规则", "chunk_index": 2, "content": "三日到账", "preview": "三日到账"}},
    ]
    judgements = [
        {
            "duplicate_rules": [
                {"topic": "退货期限", "reason": "规则相同", "old_rule": "七日", "new_rule": "七天"},
                {"topic": "退货条件", "reason": "条件相同", "old_rule": "商品完好", "new_rule": "商品保持完好"},
            ],
            "conflict_rules": [],
            "novel_rules": [{"topic": "申请入口", "new_rule": "可在APP申请", "reason": "旧知识未覆盖"}],
        },
        {
            "duplicate_rules": [],
            "conflict_rules": [
                {"topic": "退款到账", "reason": "期限不同", "old_rule": "三日", "new_rule": "七日"},
            ],
            "novel_rules": [],
        },
    ]
    database = Mock()
    database.get_document.return_value = document
    database.get_chunks_by_doc.return_value = chunks
    database.get_active_document_ids.return_value = [1, 2]
    with patch("workflows.inspection_agent.mysql_client", database), \
         patch("workflows.inspection_agent.vector_store.load_pending_embeddings", return_value=np.ones((2, 2))), \
         patch("workflows.inspection_agent.vector_store.find_active_matches_by_vectors", return_value=candidates), \
         patch.object(agent, "_judge_relations_with_llm", return_value={0: judgements[0], 1: judgements[1]}):
        report = agent.inspect_pending_document(7)

    assert report["summary"]["duplicate_count"] == 2
    assert report["summary"]["conflict_count"] == 1
    assert report["summary"]["novel_count"] == 1
    assert len(report["duplicate_rules"]) == 2
    assert report["conflict_rules"][0]["old_document_id"] == 2
    assert report["conflict_rules"][0]["cosine_similarity"] == 0.93
    assert report["summary"]["inspection_has_issue"] is True
    assert database.upsert_inspection.call_args_list[-1].args[1] == "COMPLETED"


def test_same_pair_can_contain_duplicate_and_novel_rules():
    agent = InspectionAgent()
    candidate = {"new_chunk_index": 0, "similarity": 0.91, "old": {"doc_id": 1, "source": "规则", "chunk_index": 0, "content": "普通商品支持退货", "preview": "普通商品支持退货"}}
    judgement = {
        "duplicate_rules": [
            {"topic": "退货期限", "old_rule": "七天退货", "new_rule": "7日退货", "reason": "规则相同"},
        ],
        "conflict_rules": [],
        "novel_rules": [
            {"topic": "申请方式", "new_rule": "支持APP申请", "reason": "旧知识未覆盖"},
        ],
    }
    with patch.object(agent, "_judge_relations_with_llm", return_value={0: judgement}):
        result = agent._judge_pending_candidates(
            {"id": 2, "filename": "生鲜规则"},
            [{"chunk_index": 0, "chunk_text": "生鲜商品不支持无理由退货"}],
            [candidate],
        )
    assert len(result["duplicate_rules"]) == 1
    assert len(result["novel_rules"]) == 1
    assert result["conflict_rules"] == []


def test_rule_aggregation_deduplicates_same_duplicate_from_multiple_pairs():
    agent = InspectionAgent()
    candidates = [
        {"new_chunk_index": 0, "similarity": 0.95, "old": {"doc_id": 1, "source": "规则A", "chunk_index": 0, "content": "七天退货", "preview": "七天退货"}},
        {"new_chunk_index": 0, "similarity": 0.91, "old": {"doc_id": 2, "source": "规则B", "chunk_index": 1, "content": "七日退货", "preview": "七日退货"}},
    ]
    duplicate = {
        "duplicate_rules": [{"topic": "退货期限", "old_rule": "七天退货", "new_rule": "7 天退货。", "reason": "相同"}],
        "conflict_rules": [],
        "novel_rules": [],
    }
    with patch.object(agent, "_judge_relations_with_llm", return_value={0: duplicate, 1: duplicate}):
        result = agent._judge_pending_candidates(
            {"id": 3, "filename": "新规则"},
            [{"id": 30, "chunk_index": 0, "chunk_text": "7天退货"}],
            candidates,
        )
    assert len(result["duplicate_rules"]) == 1
    assert result["duplicate_rules"][0]["old_document_id"] == 1
    assert result["duplicate_rules"][0]["new_chunk_id"] == 30


def test_old_relation_response_is_accepted_without_crashing():
    parsed = InspectionAgent._parse_relation_batch(
        '{"items":[{"pair_id":0,"relation":"conflict","topic":"退款时限",'
        '"old_rule":"24小时","new_rule":"48小时","reason":"期限不同"}]}'
    )
    assert len(parsed[0]["conflict_rules"]) == 1
    assert parsed[0]["duplicate_rules"] == []


def test_rule_judge_pure_duplicate_document_counts_duplicate_rules_only():
    agent = InspectionAgent()
    candidate = {
        "new_chunk_index": 0,
        "similarity": 0.97,
        "old": {"doc_id": 1, "source": "旧制度.txt", "chunk_index": 0, "content": "七天退货；24小时审核", "preview": "七天退货；24小时审核"},
    }
    judgement = {
        "duplicate_rules": [
            {"topic": "退货期限", "old_rule": "七天内可退货", "new_rule": "7日内可退货", "reason": "含义一致"},
            {"topic": "审核时限", "old_rule": "24小时内审核", "new_rule": "一天内完成审核", "reason": "含义一致"},
        ],
        "conflict_rules": [],
        "novel_rules": [],
    }
    with patch.object(agent, "_judge_relations_with_llm", return_value={0: judgement}):
        result = agent._judge_pending_candidates(
            {"id": 8, "filename": "新制度.txt"},
            [{"chunk_index": 0, "chunk_text": "7日内可退货；一天内完成审核"}],
            [candidate],
        )
    assert len(result["duplicate_rules"]) == 2
    assert result["conflict_rules"] == []


def test_rule_judge_pure_conflict_document_counts_conflict_rules_only():
    agent = InspectionAgent()
    candidate = {
        "new_chunk_index": 0,
        "similarity": 0.94,
        "old": {"doc_id": 1, "source": "旧制度.txt", "chunk_index": 0, "content": "七天退货；24小时退款", "preview": "七天退货；24小时退款"},
    }
    judgement = {
        "duplicate_rules": [],
        "conflict_rules": [
            {"topic": "无理由退货期限", "old_rule": "7天无理由退货", "new_rule": "14天无理由退货", "reason": "期限冲突"},
            {"topic": "退款处理时限", "old_rule": "24小时内退款", "new_rule": "48小时内退款", "reason": "时限冲突"},
        ],
        "novel_rules": [],
    }
    with patch.object(agent, "_judge_relations_with_llm", return_value={0: judgement}):
        result = agent._judge_pending_candidates(
            {"id": 9, "filename": "新制度.txt"},
            [{"chunk_index": 0, "chunk_text": "14天无理由退货；48小时内退款"}],
            [candidate],
        )
    assert result["duplicate_rules"] == []
    assert len(result["conflict_rules"]) == 2


def test_malformed_pair_is_skipped_and_reported_as_failed_pair():
    agent = InspectionAgent()
    candidates = [
        {"new_chunk_index": 0, "similarity": 0.95, "old": {"doc_id": 1, "source": "A", "chunk_index": 0, "content": "规则A", "preview": "规则A"}},
        {"new_chunk_index": 0, "similarity": 0.92, "old": {"doc_id": 2, "source": "B", "chunk_index": 1, "content": "规则B", "preview": "规则B"}},
    ]
    valid = {"duplicate_rules": [], "conflict_rules": [], "novel_rules": []}
    with patch.object(agent, "_judge_relations_with_llm", return_value={0: valid, 1: {"duplicate_rules": "错误格式"}}):
        result = agent._judge_pending_candidates(
            {"id": 10, "filename": "新制度.txt"},
            [{"chunk_index": 0, "chunk_text": "新规则"}],
            candidates,
        )
    assert result["failed_pair_ids"] == [1]


def test_only_novel_rules_do_not_mark_inspection_as_issue():
    agent = InspectionAgent()
    database = Mock()
    database.get_document.return_value = {"id": 12, "status": "PENDING_REVIEW", "filename": "新增规则.txt"}
    database.get_chunks_by_doc.return_value = [{"chunk_index": 0, "chunk_text": "新增APP申请入口" * 20}]
    database.get_active_document_ids.return_value = [1]
    candidate = {"new_chunk_index": 0, "similarity": 0.9, "old": {"doc_id": 1, "source": "旧规则", "chunk_index": 0, "content": "线下申请", "preview": "线下申请"}}
    judgement = {
        "duplicate_rules": [],
        "conflict_rules": [],
        "novel_rules": [{"topic": "申请入口", "new_rule": "支持APP申请", "reason": "旧知识未覆盖"}],
    }
    with patch("workflows.inspection_agent.mysql_client", database), \
         patch("workflows.inspection_agent.vector_store.load_pending_embeddings", return_value=np.ones((1, 2))), \
         patch("workflows.inspection_agent.vector_store.find_active_matches_by_vectors", return_value=[candidate]), \
         patch.object(agent, "_judge_relations_with_llm", return_value={0: judgement}):
        report = agent.inspect_pending_document(12)
    assert report["summary"]["inspection_has_issue"] is False
    assert report["summary"]["novel_count"] == 1
    assert database.upsert_inspection.call_args_list[-1].args[1] == "COMPLETED"


def test_llm_judge_failure_marks_inspection_failed_and_never_publishes():
    agent = InspectionAgent()
    database = Mock()
    database.get_document.return_value = {"id": 7, "status": "PENDING_REVIEW", "filename": "faq.txt"}
    database.get_chunks_by_doc.return_value = [{"chunk_index": 0, "chunk_text": "退货规则" * 30}]
    database.get_active_document_ids.return_value = [1]
    candidate = {"new_chunk_index": 0, "similarity": 0.95, "old": {"doc_id": 1, "source": "旧规则", "chunk_index": 0, "content": "旧规则", "preview": "旧规则"}}
    with patch("workflows.inspection_agent.mysql_client", database), \
         patch("workflows.inspection_agent.vector_store.load_pending_embeddings", return_value=np.ones((1, 2))), \
         patch("workflows.inspection_agent.vector_store.find_active_matches_by_vectors", return_value=[candidate]), \
         patch.object(agent, "_judge_relations_with_llm", side_effect=TimeoutError("LLM超时")):
        report = agent.inspect_pending_document(7)
    assert report["summary"]["judge_complete"] is False
    assert report["judge_failed_pair_ids"] == [0]
    assert database.upsert_inspection.call_args_list[-1].args[1] == "FAILED"


def test_upload_finishes_as_pending_and_does_not_add_online_vectors(tmp_path, monkeypatch):
    monkeypatch.setattr(document_routes, "_table_ready", True)
    monkeypatch.setattr(document_routes, "UPLOAD_DIR", str(tmp_path))
    chunks = [Document(page_content="企业规则" * 30, metadata={"chunk_index": 0})]
    database = Mock()
    database.create_document.return_value = 21
    database.insert_chunks.return_value = 1
    store = Mock()
    store.create_pending_embeddings.return_value = np.ones((1, 2))
    upload = UploadFile(filename="规则.txt", file=io.BytesIO("企业规则".encode("utf-8")))
    background = BackgroundTasks()
    with patch.object(document_routes, "mysql_client", database), \
         patch.object(document_routes, "vector_store", store), \
         patch.object(document_routes.parser, "parse", return_value=chunks):
        result = asyncio.run(document_routes.upload_document(background, upload, "规则", {"is_admin": True}))

    assert result["document_status"] == "PENDING_REVIEW"
    database.update_document_status.assert_called_with(21, "PENDING_REVIEW", 1)
    store.add_documents.assert_not_called()


def test_approve_is_idempotent_and_does_not_duplicate_vectors():
    database = Mock()
    database.get_document.return_value = {"id": 5, "status": "ACTIVE"}
    store = Mock()
    with patch.object(document_routes, "mysql_client", database), patch.object(document_routes, "vector_store", store):
        result = document_routes._approve_document(5)
    assert result["document_status"] == "ACTIVE"
    store.publish_precomputed_documents.assert_not_called()


def test_pending_document_becomes_active_only_after_successful_publish():
    database = Mock()
    database.get_document.return_value = {
        "id": 6,
        "status": "PENDING_REVIEW",
        "filename": "新规则.txt",
    }
    database.get_inspection.return_value = {"inspection_status": "COMPLETED"}
    database.get_chunks_by_doc.return_value = [
        {"chunk_index": 0, "chunk_text": "审核通过后发布的规则"}
    ]
    store = Mock()
    store.contains_document.return_value = False
    store.load_pending_embeddings.return_value = np.ones((1, 2), dtype=np.float32)
    with patch.object(document_routes, "mysql_client", database), patch.object(document_routes, "vector_store", store):
        result = document_routes._approve_document(6)
    store.publish_precomputed_documents.assert_called_once()
    database.update_document_status.assert_called_once_with(6, "ACTIVE", 1)
    store.delete_pending_embeddings.assert_called_once_with(6)
    assert result["document_status"] == "ACTIVE"


def test_reject_keeps_document_out_of_online_faiss(monkeypatch):
    monkeypatch.setattr(document_routes, "_table_ready", True)
    database = Mock()
    database.get_document.return_value = {"id": 8, "status": "PENDING_REVIEW", "chunks_count": 2}
    store = Mock()
    with patch.object(document_routes, "mysql_client", database), patch.object(document_routes, "vector_store", store):
        result = asyncio.run(document_routes.reject_document(
            8,
            document_routes.RejectRequest(reason="规则冲突"),
            {"is_admin": True},
        ))
    assert result["document_status"] == "REJECTED"
    database.update_document_status.assert_called_with(8, "REJECTED", 2)
    store.publish_precomputed_documents.assert_not_called()


def test_low_quality_rule_is_preserved_for_pending_chunks():
    issues = InspectionAgent._inspect_pending_chunk_quality([
        {"chunk_index": 0, "chunk_text": "短"},
        {"chunk_index": 1, "chunk_text": "合格内容" * 40},
    ])
    assert [item["chunk_index"] for item in issues] == [0]
