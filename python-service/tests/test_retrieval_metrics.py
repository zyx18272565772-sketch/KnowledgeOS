import math

import pytest

from evaluation.metrics import calculate_metrics, make_document_key, mean_metrics


def test_make_document_key_normalizes_json_and_metadata_values():
    assert make_document_key(33, "1") == ("33", 1)


def test_perfect_ranking():
    relevant = [("33", 0), ("33", 1)]
    result = calculate_metrics([("33", 0), ("33", 1)], relevant, k=2)

    assert result.precision_at_k == 1.0
    assert result.recall_at_k == 1.0
    assert result.hit_rate_at_k == 1.0
    assert result.mrr_at_k == 1.0
    assert result.ndcg_at_k == 1.0


def test_partial_ranking_uses_fixed_precision_denominator():
    relevant = [("33", 0), ("33", 1)]
    retrieved = [("99", 0), ("33", 0)]
    result = calculate_metrics(retrieved, relevant, k=5)

    assert result.precision_at_k == pytest.approx(0.2)
    assert result.recall_at_k == pytest.approx(0.5)
    assert result.hit_rate_at_k == 1.0
    assert result.mrr_at_k == pytest.approx(0.5)
    expected_dcg = 1 / math.log2(3)
    ideal_dcg = 1 / math.log2(2) + 1 / math.log2(3)
    assert result.ndcg_at_k == pytest.approx(expected_dcg / ideal_dcg)


def test_duplicate_result_only_receives_credit_once():
    relevant = [("33", 0)]
    result = calculate_metrics(
        [("33", 0), ("33", 0), ("99", 0)],
        relevant,
        k=3,
    )

    assert result.precision_at_k == pytest.approx(1 / 3)
    assert result.recall_at_k == 1.0
    assert result.ndcg_at_k == 1.0


def test_no_hit():
    result = calculate_metrics(
        [("99", 0), ("98", 0)],
        [("33", 0)],
        k=2,
    )

    assert result.precision_at_k == 0.0
    assert result.recall_at_k == 0.0
    assert result.hit_rate_at_k == 0.0
    assert result.mrr_at_k == 0.0
    assert result.ndcg_at_k == 0.0


def test_invalid_arguments():
    with pytest.raises(ValueError):
        calculate_metrics([], [("33", 0)], k=0)
    with pytest.raises(ValueError):
        calculate_metrics([], [], k=5)


def test_mean_metrics():
    first = calculate_metrics([("33", 0)], [("33", 0)], k=1)
    second = calculate_metrics([("99", 0)], [("33", 0)], k=1)
    summary = mean_metrics([first, second])

    assert summary["precision_at_k"] == 0.5
    assert summary["recall_at_k"] == 0.5
    assert summary["hit_rate_at_k"] == 0.5
    assert summary["mrr_at_k"] == 0.5
    assert summary["ndcg_at_k"] == 0.5
