"""Compare direct FAISS retrieval with the configured reranking pipeline."""

from __future__ import annotations

import argparse
import json
import os
import statistics
import time
from collections import Counter
from datetime import datetime
from pathlib import Path
from typing import Any

from dotenv import load_dotenv

from evaluation.metrics import (
    RetrievalMetrics,
    calculate_metrics,
    make_document_key,
    mean_metrics,
)


PROJECT_ROOT = Path(__file__).resolve().parents[1]
DEFAULT_DATASET = Path(__file__).with_name("retrieval_dataset.jsonl")


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description="项目1知识库检索离线评测")
    parser.add_argument("--dataset", type=Path, default=DEFAULT_DATASET)
    parser.add_argument("--output-dir", type=Path)
    parser.add_argument("--top-k", type=int, default=5)
    parser.add_argument("--limit", type=int)
    return parser.parse_args()


def load_dataset(path: Path, limit: int | None = None) -> list[dict[str, Any]]:
    cases: list[dict[str, Any]] = []
    seen_ids: set[str] = set()
    with path.open("r", encoding="utf-8") as handle:
        for line_number, line in enumerate(handle, start=1):
            line = line.strip()
            if not line:
                continue
            case = json.loads(line)
            case_id = str(case.get("id", "")).strip()
            question = str(case.get("question", "")).strip()
            relevant = case.get("relevant_chunks") or []
            if not case_id or not question or not relevant:
                raise ValueError(f"{path}:{line_number} 缺少 id/question/relevant_chunks")
            if case_id in seen_ids:
                raise ValueError(f"{path}:{line_number} id 重复: {case_id}")
            seen_ids.add(case_id)
            for ref in relevant:
                make_document_key(ref["doc_id"], ref["chunk_index"])
            cases.append(case)
            if limit is not None and len(cases) >= limit:
                break
    if not cases:
        raise ValueError("评测集为空")
    return cases


def document_key(document: Any) -> tuple[str, int] | None:
    metadata = getattr(document, "metadata", {}) or {}
    if metadata.get("doc_id") is None or metadata.get("chunk_index") is None:
        return None
    try:
        return make_document_key(metadata["doc_id"], metadata["chunk_index"])
    except (TypeError, ValueError):
        return None


def serialize_document(
    document: Any,
    rank: int,
    include_rerank_score: bool,
) -> dict[str, Any]:
    metadata = dict(getattr(document, "metadata", {}) or {})
    key = document_key(document)
    return {
        "rank": rank,
        "key": {"doc_id": key[0], "chunk_index": key[1]} if key else None,
        "source": metadata.get("source"),
        "rerank_score": metadata.get("rerank_score") if include_rerank_score else None,
        "preview": str(getattr(document, "page_content", ""))[:160],
    }


def inspect_index(vector_store: Any, cases: list[dict[str, Any]]) -> dict[str, Any]:
    store = getattr(vector_store, "vector_store", None)
    docstore = getattr(store, "docstore", None)
    documents = list(getattr(docstore, "_dict", {}).values())

    expected_keys = {
        make_document_key(ref["doc_id"], ref["chunk_index"])
        for case in cases
        for ref in case["relevant_chunks"]
    }
    existing_keys = {key for doc in documents if (key := document_key(doc)) is not None}
    source_counts = Counter(
        str((getattr(doc, "metadata", {}) or {}).get("source", "unknown"))
        for doc in documents
    )

    replacement_chars = 0
    total_chars = 0
    corrupted_documents = 0
    for doc in documents:
        text = str(getattr(doc, "page_content", ""))
        count = text.count("\ufffd")
        replacement_chars += count
        total_chars += len(text)
        if text and count / len(text) >= 0.01:
            corrupted_documents += 1

    missing = sorted(expected_keys - existing_keys)
    enterprise_documents = sum(
        1 for doc in documents
        if isinstance((getattr(doc, "metadata", {}) or {}).get("doc_id"), int)
    )
    return {
        "total_chunks": len(documents),
        "enterprise_chunks_with_numeric_doc_id": enterprise_documents,
        "other_chunks": len(documents) - enterprise_documents,
        "source_counts": dict(source_counts),
        "replacement_character_count": replacement_chars,
        "replacement_character_ratio": replacement_chars / total_chars if total_chars else 0.0,
        "suspected_corrupted_chunks": corrupted_documents,
        "missing_labeled_chunks": [
            {"doc_id": doc_id, "chunk_index": chunk_index}
            for doc_id, chunk_index in missing
        ],
    }


def run_variant(
    vector_store: Any,
    case: dict[str, Any],
    top_k: int,
    use_rerank: bool,
) -> tuple[dict[str, Any], RetrievalMetrics]:
    started = time.perf_counter()
    documents = vector_store.search(
        query=case["question"],
        k=top_k,
        use_rerank=use_rerank,
    )
    duration_ms = (time.perf_counter() - started) * 1000

    retrieved_keys = [
        key for doc in documents
        if (key := document_key(doc)) is not None
    ]
    relevant_keys = [
        make_document_key(ref["doc_id"], ref["chunk_index"])
        for ref in case["relevant_chunks"]
    ]
    metrics = calculate_metrics(retrieved_keys, relevant_keys, top_k)
    return {
        "duration_ms": duration_ms,
        "retrieved": [
            serialize_document(doc, rank, include_rerank_score=use_rerank)
            for rank, doc in enumerate(documents, start=1)
        ],
        "metrics": metrics.to_dict(),
    }, metrics


def summarize_variant(
    results: list[dict[str, Any]],
    metric_objects: list[RetrievalMetrics],
    errors: int,
) -> dict[str, Any]:
    durations = [item["duration_ms"] for item in results]
    summary: dict[str, Any] = mean_metrics(metric_objects)
    summary.update({
        "successful_cases": len(results),
        "error_cases": errors,
        "average_duration_ms": statistics.mean(durations) if durations else 0.0,
        "p95_duration_ms": (
            sorted(durations)[max(0, int(len(durations) * 0.95) - 1)]
            if durations else 0.0
        ),
    })
    return summary


def create_report(payload: dict[str, Any], top_k: int) -> str:
    health = payload["index_health"]
    summaries = payload["summary"]
    metric_names = [
        "precision_at_k", "recall_at_k", "hit_rate_at_k",
        "mrr_at_k", "ndcg_at_k",
    ]

    lines = [
        "# 检索评测报告",
        "",
        f"- 运行时间：{payload['run_at']}",
        f"- 测试问题：{payload['case_count']} 条",
        f"- Top K：{top_k}",
        f"- 当前 Reranker：{payload['configuration']['reranker_type']}",
        f"- 候选倍率：{payload['configuration']['candidate_multiplier']}",
        "",
        "## 索引健康检查",
        "",
        f"- 索引总 Chunk：{health['total_chunks']}",
        f"- 企业文档 Chunk（数字 doc_id）：{health['enterprise_chunks_with_numeric_doc_id']}",
        f"- 其他来源 Chunk：{health['other_chunks']}",
        f"- 疑似乱码 Chunk：{health['suspected_corrupted_chunks']}",
        f"- 未在索引中找到的标注 Chunk：{len(health['missing_labeled_chunks'])}",
        "",
    ]
    if health["suspected_corrupted_chunks"]:
        lines.extend([
            "> 警告：索引中检测到大量 Unicode 替换字符，中文内容可能在入库时发生编码损坏。",
            "",
        ])
    if health["missing_labeled_chunks"]:
        lines.extend([
            "> 警告：部分人工标注 Chunk 不存在于当前索引，本次指标不能作为有效基线。",
            "",
        ])

    lines.extend([
        f"## 汇总指标（@{top_k}）",
        "",
        "| 方案 | Precision | Recall | HitRate | MRR | NDCG | 平均耗时(ms) | P95(ms) | 错误数 |",
        "|---|---:|---:|---:|---:|---:|---:|---:|---:|",
    ])
    for name, label in (
        ("vector_only", "FAISS Top 5"),
        ("with_rerank", "FAISS候选 + Rerank"),
    ):
        item = summaries[name]
        lines.append(
            f"| {label} | {item['precision_at_k']:.3f} | {item['recall_at_k']:.3f} | "
            f"{item['hit_rate_at_k']:.3f} | {item['mrr_at_k']:.3f} | "
            f"{item['ndcg_at_k']:.3f} | {item['average_duration_ms']:.1f} | "
            f"{item['p95_duration_ms']:.1f} | {item['error_cases']} |"
        )

    lines.extend(["", "## 重排序变化", ""])
    for metric in metric_names:
        lines.append(f"- {metric}: {summaries['delta'][metric]:+.3f}")
    lines.append(
        f"- average_duration_ms: {summaries['delta']['average_duration_ms']:+.1f}"
    )

    lines.extend(["", "## 未命中问题", ""])
    misses = []
    for case in payload["cases"]:
        for variant_name, label in (
            ("vector_only", "FAISS"),
            ("with_rerank", "Rerank"),
        ):
            variant = case.get(variant_name)
            if variant and not variant.get("error") and variant["metrics"]["hit_rate_at_k"] == 0:
                misses.append(f"- [{label}] {case['id']}：{case['question']}")
    lines.extend(misses or ["- 无"])
    lines.append("")
    return "\n".join(lines)


def main() -> int:
    args = parse_args()
    if args.top_k <= 0:
        raise ValueError("--top-k 必须大于 0")

    load_dotenv(PROJECT_ROOT / ".env")
    os.chdir(PROJECT_ROOT)

    # Import after loading .env because core.config initializes at import time.
    from core.config import config
    from core.vector_store import vector_store

    cases = load_dataset(args.dataset.resolve(), args.limit)
    health = inspect_index(vector_store, cases)

    # Warm up embedding inference and both search paths before measuring latency.
    # Otherwise the first vector-only case pays model/index cold-start cost and
    # makes the reranked path incorrectly look faster.
    warmup_query = cases[0]["question"]
    vector_store.search(query=warmup_query, k=args.top_k, use_rerank=False)
    vector_store.search(query=warmup_query, k=args.top_k, use_rerank=True)

    output_dir = args.output_dir
    if output_dir is None:
        timestamp = datetime.now().strftime("%Y%m%d_%H%M%S")
        output_dir = Path(__file__).parent / "results" / timestamp
    output_dir = output_dir.resolve()
    output_dir.mkdir(parents=True, exist_ok=True)

    variant_metrics: dict[str, list[RetrievalMetrics]] = {
        "vector_only": [],
        "with_rerank": [],
    }
    variant_results: dict[str, list[dict[str, Any]]] = {
        "vector_only": [],
        "with_rerank": [],
    }
    error_counts = Counter()
    case_results: list[dict[str, Any]] = []

    for index, case in enumerate(cases, start=1):
        print(f"[{index}/{len(cases)}] {case['id']} {case['question']}")
        case_result = {
            "id": case["id"],
            "question": case["question"],
            "category": case.get("category"),
            "relevant_chunks": case["relevant_chunks"],
        }
        for name, use_rerank in (("vector_only", False), ("with_rerank", True)):
            try:
                result, metrics = run_variant(
                    vector_store, case, args.top_k, use_rerank
                )
                case_result[name] = result
                variant_results[name].append(result)
                variant_metrics[name].append(metrics)
            except Exception as exc:
                error_counts[name] += 1
                case_result[name] = {"error": f"{type(exc).__name__}: {exc}"}
        case_results.append(case_result)

    summaries = {
        name: summarize_variant(
            variant_results[name],
            variant_metrics[name],
            error_counts[name],
        )
        for name in variant_metrics
    }
    delta_keys = [
        "precision_at_k", "recall_at_k", "hit_rate_at_k",
        "mrr_at_k", "ndcg_at_k", "average_duration_ms",
    ]
    summaries["delta"] = {
        key: summaries["with_rerank"][key] - summaries["vector_only"][key]
        for key in delta_keys
    }

    payload = {
        "run_at": datetime.now().isoformat(timespec="seconds"),
        "case_count": len(cases),
        "configuration": {
            "top_k": args.top_k,
            "reranker_type": config.RERANKER_TYPE,
            "candidate_multiplier": config.RETRIEVAL_CANDIDATE_MULTIPLIER,
            "embedding_model": config.EMBEDDING_MODEL,
            "vector_store": "milvus" if vector_store.use_milvus else "faiss",
        },
        "index_health": health,
        "summary": summaries,
        "cases": case_results,
    }

    json_path = output_dir / "results.json"
    report_path = output_dir / "report.md"
    json_path.write_text(
        json.dumps(payload, ensure_ascii=False, indent=2),
        encoding="utf-8",
    )
    report_path.write_text(create_report(payload, args.top_k), encoding="utf-8")
    print(f"\nJSON结果：{json_path}")
    print(f"Markdown报告：{report_path}")
    return int(any(error_counts.values()))


if __name__ == "__main__":
    raise SystemExit(main())
