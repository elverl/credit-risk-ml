"""Run five initial checks, then all remaining golden evalset cases on Groq."""

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

from credit_risk.rag.pipeline import (  # noqa: E402
    RAGPipeline,
    enforce_grounding_guardrails,
    ground_answer_sources,
)
from credit_risk.rag.retriever import DenseRetriever  # noqa: E402


INITIAL_IDS = ("ft-001", "mm-002", "sh-001", "oa-001", "ip-002")


def _answer_with_retry(pipeline, case, attempts=6):
    for attempt in range(1, attempts + 1):
        try:
            return pipeline.answer(case["question"], case.get("model_context"))
        except requests.HTTPError as exc:
            if exc.response is None or exc.response.status_code != 429 or attempt == attempts:
                raise
            retry_after = exc.response.headers.get("Retry-After")
            try:
                delay = float(retry_after) if retry_after else 15.0 * attempt
            except ValueError:
                delay = 15.0 * attempt
            delay = min(max(delay, 1.0), 45.0)
            print(f"Groq rate limit; retrying {case['id']} in {delay:.0f}s.")
            time.sleep(delay)


def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--rerun-ids", nargs="*", default=[], help="replace selected saved cases")
    args = parser.parse_args()
    if not os.getenv("GROQ_API_KEY"):
        raise SystemExit("GROQ_API_KEY is not configured; no remote call was made.")
    rag_dir = ROOT / "artifacts" / "rag"
    cases = [
        json.loads(line)
        for line in (ROOT / "evaluation" / "rag_evalset_v1.jsonl").read_text(encoding="utf-8").splitlines()
        if line.strip()
    ]
    by_id = {case["id"]: case for case in cases}
    ordered = [by_id[case_id] for case_id in INITIAL_IDS]
    ordered.extend(case for case in cases if case["id"] not in INITIAL_IDS)
    retriever = DenseRetriever.load_or_build(rag_dir / "kb_chunks.jsonl", rag_dir)
    pipeline = RAGPipeline(retriever)
    output_path = rag_dir / "rag_responses_v1.jsonl"
    results = (
        [json.loads(line) for line in output_path.read_text(encoding="utf-8").splitlines() if line.strip()]
        if output_path.exists() else []
    )
    if args.rerun_ids:
        requested = set(args.rerun_ids)
        unknown = requested - set(by_id)
        if unknown:
            raise SystemExit(f"Unknown case IDs: {sorted(unknown)}")
        results = [row for row in results if row["id"] not in requested]
    for row in results:
        guarded = enforce_grounding_guardrails(
            row["query"], row["generated_answer"], row.get("model_context")
        )
        row["generated_answer"] = ground_answer_sources(guarded, row["retrieved_chunk_ids"])
    completed = {row["id"] for row in results if row.get("generation_success")}
    try:
        pending = [case for case in ordered if case["id"] not in completed]
        for index, case in enumerate(pending, start=len(completed) + 1):
            result = _answer_with_retry(pipeline, case).to_dict()
            result.update({"id": case["id"], "category": case["category"],
                           "initial_validation": case["id"] in INITIAL_IDS,
                           "generation_success": result.get("finish_reason") != "length"})
            results.append(result)
            # Persist after every successful call so interrupted runs remain auditable.
            output_path.write_text(
                "".join(json.dumps(row, ensure_ascii=False, separators=(",", ":")) + "\n" for row in results),
                encoding="utf-8",
            )
            print(f"[{index:02d}/24] {case['id']} OK ({result['latency_seconds']:.2f}s)")
            time.sleep(1.0)
    except Exception as exc:
        print(f"Generation stopped after {len(results)} successes: {type(exc).__name__}: {exc}")
        raise

    output_path.write_text(
        "".join(json.dumps(row, ensure_ascii=False, separators=(",", ":")) + "\n" for row in results),
        encoding="utf-8",
    )

    token_totals: dict[str, int] = {}
    for row in results:
        for key, value in (row.get("usage") or {}).items():
            if isinstance(value, int):
                token_totals[key] = token_totals.get(key, 0) + value
    summary = {
        "provider": "Groq",
        "model": results[0]["model_name"] if results else "openai/gpt-oss-20b",
        "cases": len(results),
        "successes": sum(row["generation_success"] for row in results),
        "failures": sum(not row["generation_success"] for row in results),
        "initial_validation_ids": list(INITIAL_IDS),
        "average_latency_seconds": sum(row["latency_seconds"] for row in results) / len(results),
        "token_totals": token_totals or None,
    }
    (rag_dir / "rag_generation_summary_v1.json").write_text(
        json.dumps(summary, ensure_ascii=False, indent=2) + "\n", encoding="utf-8"
    )
    print(json.dumps(summary, ensure_ascii=False, indent=2))


if __name__ == "__main__":
    main()
