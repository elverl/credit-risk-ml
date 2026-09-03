"""Build/reuse dense embeddings and evaluate Top-3 retrieval."""

from __future__ import annotations

import json
import sys
from pathlib import Path


ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT / "src"))

from credit_risk.rag.retriever import (  # noqa: E402
    DEFAULT_DIMENSION,
    EMBEDDING_MODEL,
    DenseRetriever,
    evaluate_results,
)


def _write_jsonl(path: Path, rows: list[dict[str, object]]) -> None:
    path.write_text(
        "".join(json.dumps(row, ensure_ascii=False, separators=(",", ":")) + "\n" for row in rows),
        encoding="utf-8",
    )


def main() -> None:
    rag_dir = ROOT / "artifacts" / "rag"
    corpus_path = rag_dir / "kb_chunks.jsonl"
    evalset_path = ROOT / "evaluation" / "rag_evalset_v1.jsonl"
    retriever = DenseRetriever.load_or_build(corpus_path, rag_dir, DEFAULT_DIMENSION)
    cases = [json.loads(line) for line in evalset_path.read_text(encoding="utf-8").splitlines() if line.strip()]
    results: list[dict[str, object]] = []
    for case in cases:
        expected = set(case["expected_chunk_ids"])
        top = retriever.search(case["question"], top_k=3)
        first_rank = next((rank for rank, item in enumerate(top, start=1) if item["chunk_id"] in expected), None)
        results.append(
            {
                "id": case["id"],
                "category": case["category"],
                "question": case["question"],
                "expected_chunk_ids": case["expected_chunk_ids"],
                "hit_at_3": first_rank is not None,
                "first_correct_rank": first_rank,
                "top_3": top,
            }
        )
    metrics = evaluate_results(results)
    summary = {
        "embedding_model": EMBEDDING_MODEL,
        "embedding_dimension": int(retriever.embeddings.shape[1]),
        "similarity": "cosine",
        "top_k": 3,
        "chunk_count": len(retriever.chunks),
        **metrics,
        "miss_ids": [row["id"] for row in results if not row["hit_at_3"]],
    }
    _write_jsonl(rag_dir / "retrieval_results.jsonl", results)
    (rag_dir / "retrieval_eval.json").write_text(
        json.dumps(summary, ensure_ascii=False, indent=2) + "\n", encoding="utf-8"
    )
    print(json.dumps(summary, ensure_ascii=False, indent=2))


if __name__ == "__main__":
    main()
