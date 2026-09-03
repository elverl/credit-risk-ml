"""Generate the no-retrieval baseline and score Baseline vs RAG."""

from __future__ import annotations

import argparse
import json
import os
import sys
import time
from pathlib import Path

import requests


ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT / "src"))

from credit_risk.rag.comparison import (  # noqa: E402
    BaselinePipeline,
    aggregate_comparison,
    evaluate_answer,
)


def write_jsonl(path: Path, rows: list[dict]) -> None:
    path.write_text("".join(json.dumps(row, ensure_ascii=False, separators=(",", ":")) + "\n" for row in rows), encoding="utf-8")


def call_with_retry(pipeline, case, attempts=6):
    for attempt in range(1, attempts + 1):
        try:
            return pipeline.answer(case["question"], case.get("model_context"))
        except requests.HTTPError as exc:
            if exc.response is None or exc.response.status_code != 429 or attempt == attempts:
                raise
            header = exc.response.headers.get("Retry-After")
            try:
                delay = float(header) if header else 15.0 * attempt
            except ValueError:
                delay = 15.0 * attempt
            delay = min(max(delay, 1.0), 45.0)
            print(f"Groq rate limit; retrying {case['id']} in {delay:.0f}s.")
            time.sleep(delay)


def operational(rows: list[dict]) -> dict:
    totals: dict[str, int] = {}
    for row in rows:
        for key, value in (row.get("usage") or {}).items():
            if isinstance(value, int):
                totals[key] = totals.get(key, 0) + value
    return {
        "cases": len(rows),
        "successes": sum(bool(row.get("generation_success", True)) for row in rows),
        "failures": sum(not bool(row.get("generation_success", True)) for row in rows),
        "average_latency_seconds": sum(row["latency_seconds"] for row in rows) / len(rows),
        "token_totals": totals or None,
    }


def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--score-only", action="store_true")
    args = parser.parse_args()
    rag_dir = ROOT / "artifacts" / "rag"
    baseline_path = rag_dir / "baseline_responses_v1.jsonl"
    cases = [json.loads(line) for line in (ROOT / "evaluation" / "rag_evalset_v1.jsonl").read_text(encoding="utf-8").splitlines() if line.strip()]
    baseline = [json.loads(line) for line in baseline_path.read_text(encoding="utf-8").splitlines() if line.strip()] if baseline_path.exists() else []
    completed = {row["id"] for row in baseline}
    if not args.score_only and len(completed) < len(cases):
        if not os.getenv("GROQ_API_KEY"):
            raise SystemExit("GROQ_API_KEY is not configured; no remote call was made.")
        pipeline = BaselinePipeline()
        for case in cases:
            if case["id"] in completed:
                continue
            result = call_with_retry(pipeline, case).to_dict()
            result.update({"id": case["id"], "category": case["category"],
                           "generation_success": result.get("finish_reason") != "length"})
            baseline.append(result)
            write_jsonl(baseline_path, baseline)
            print(f"[{len(baseline):02d}/24] {case['id']} OK ({result['latency_seconds']:.2f}s)")
            time.sleep(1.0)
    if len(baseline) != len(cases):
        raise SystemExit(f"Baseline incomplete: {len(baseline)}/24")

    rag = [json.loads(line) for line in (rag_dir / "rag_responses_v1.jsonl").read_text(encoding="utf-8").splitlines() if line.strip()]
    by_baseline = {row["id"]: row for row in baseline}
    by_rag = {row["id"]: row for row in rag}
    chunks = {row["chunk_id"]: row for row in [json.loads(line) for line in (rag_dir / "kb_chunks.jsonl").read_text(encoding="utf-8").splitlines() if line.strip()]}
    results = []
    for case in cases:
        baseline_response = by_baseline[case["id"]]
        rag_response = by_rag[case["id"]]
        model_context = json.dumps(case.get("model_context"), ensure_ascii=False) if case.get("model_context") else ""
        rag_context = "\n".join(chunks[identifier]["text"] for identifier in rag_response["retrieved_chunk_ids"])
        row = {
            "id": case["id"],
            "category": case["category"],
            "retrieval_hit_at_3": bool(set(case["expected_chunk_ids"]) & set(rag_response["retrieved_chunk_ids"])),
            "baseline": evaluate_answer(case, baseline_response, model_context),
            "rag": evaluate_answer(case, rag_response, rag_context + "\n" + model_context),
        }
        base_quality = sum(row["baseline"][key] for key in ("factual_correctness", "groundedness", "answer_relevance", "abstention_accuracy"))
        rag_quality = sum(row["rag"][key] for key in ("factual_correctness", "groundedness", "answer_relevance", "abstention_accuracy"))
        row["outcome"] = "improved" if rag_quality > base_quality + 1e-9 else "worsened" if rag_quality < base_quality - 1e-9 else "tied"
        results.append(row)
    summary = aggregate_comparison(results)
    summary["design"] = {"model": "openai/gpt-oss-20b", "cases": 24, "rag_top_k": 3,
                         "experimental_variable": "retrieved knowledge context"}
    summary["operational"] = {"baseline": operational(baseline), "rag": operational(rag)}
    summary["outcomes"] = {name: [row["id"] for row in results if row["outcome"] == name]
                           for name in ("improved", "tied", "worsened")}
    hits = [row for row in results if row["retrieval_hit_at_3"]]
    misses = [row for row in results if not row["retrieval_hit_at_3"]]
    summary["retrieval_effect"] = {
        "hit_cases": len(hits), "miss_cases": len(misses),
        "rag_factual_on_hits": sum(row["rag"]["factual_correctness"] for row in hits) / len(hits),
        "rag_factual_on_misses": sum(row["rag"]["factual_correctness"] for row in misses) / len(misses),
    }
    write_jsonl(rag_dir / "rag_comparison_results_v1.jsonl", results)
    (rag_dir / "rag_comparison_summary_v1.json").write_text(json.dumps(summary, ensure_ascii=False, indent=2) + "\n", encoding="utf-8")
    print(json.dumps(summary, ensure_ascii=False, indent=2))


if __name__ == "__main__":
    main()
