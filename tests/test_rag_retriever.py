"""Tests for local dense retrieval and retrieval metrics."""

import json
from pathlib import Path

import numpy as np
import pytest

from credit_risk.rag.retriever import DenseRetriever, evaluate_results, load_corpus


ROOT = Path(__file__).resolve().parents[1]
CORPUS_PATH = ROOT / "artifacts" / "rag" / "kb_chunks.jsonl"


@pytest.fixture(scope="module")
def retriever():
    return DenseRetriever.fit(load_corpus(CORPUS_PATH), dimension=12)


def test_load_corpus():
    chunks = load_corpus(CORPUS_PATH)
    assert len(chunks) == 56
    assert len({chunk["chunk_id"] for chunk in chunks}) == 56


def test_embedding_dimension_and_values(retriever):
    assert retriever.embeddings.shape == (56, 12)
    assert np.isfinite(retriever.embeddings).all()
    query = retriever.embed_query("threshold de XGBoost")
    assert query.shape == (12,)
    assert np.isfinite(query).all()


def test_top_k_order_and_valid_ids(retriever):
    results = retriever.search("¿Qué significa incumplimiento?", top_k=3)
    known = {chunk["chunk_id"] for chunk in retriever.chunks}
    assert len(results) == 3
    assert [row["score"] for row in results] == sorted(
        [row["score"] for row in results], reverse=True
    )
    assert all(row["chunk_id"] in known for row in results)
    assert all(set(row) == {"chunk_id", "score", "source", "section_path", "text"} for row in results)


def test_hit_rate_and_mrr_at_3_calculation():
    rows = [
        {"category": "a", "hit_at_3": True, "first_correct_rank": 1},
        {"category": "a", "hit_at_3": True, "first_correct_rank": 2},
        {"category": "b", "hit_at_3": False, "first_correct_rank": None},
    ]
    metrics = evaluate_results(rows)
    assert metrics["hit_rate_at_3"] == pytest.approx(2 / 3)
    assert metrics["mrr_at_3"] == pytest.approx(0.5)
    assert metrics["by_category"]["a"]["hit_rate_at_3"] == 1.0
    assert metrics["by_category"]["a"]["mrr_at_3"] == pytest.approx(0.75)
