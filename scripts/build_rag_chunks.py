"""Build deterministic KB chunks and enrich the RAG evaluation set."""

from __future__ import annotations

import argparse
import json
import sys
from pathlib import Path

ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT / "src"))

from credit_risk.rag.chunking import build_chunks, resolve_expected_chunk_ids  # noqa: E402


def write_jsonl(path: Path, rows: list[dict[str, object]]) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    payload = "".join(json.dumps(row, ensure_ascii=False, separators=(",", ":")) + "\n" for row in rows)
    path.write_text(payload, encoding="utf-8")


def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--check", action="store_true", help="validate without writing files")
    args = parser.parse_args()
    chunks_path = ROOT / "artifacts" / "rag" / "kb_chunks.jsonl"
    evalset_path = ROOT / "evaluation" / "rag_evalset_v1.jsonl"
    chunks = build_chunks(ROOT / "docs" / "knowledge_base")
    records = [json.loads(line) for line in evalset_path.read_text(encoding="utf-8").splitlines() if line.strip()]
    for record in records:
        resolved = resolve_expected_chunk_ids(record, chunks)
        if resolved:
            record["expected_chunk_ids"] = resolved
        else:
            record.pop("expected_chunk_ids", None)
    known_ids = {chunk["chunk_id"] for chunk in chunks}
    referenced = {identifier for record in records for identifier in record.get("expected_chunk_ids", [])}
    if not referenced <= known_ids:
        raise ValueError(f"Broken expected chunk IDs: {sorted(referenced - known_ids)}")
    if args.check:
        expected_chunks = "".join(json.dumps(row, ensure_ascii=False, separators=(",", ":")) + "\n" for row in chunks)
        expected_evalset = "".join(json.dumps(row, ensure_ascii=False, separators=(",", ":")) + "\n" for row in records)
        if not chunks_path.exists() or chunks_path.read_text(encoding="utf-8") != expected_chunks:
            raise SystemExit("kb_chunks.jsonl is stale; run scripts/build_rag_chunks.py")
        if evalset_path.read_text(encoding="utf-8") != expected_evalset:
            raise SystemExit("rag_evalset_v1.jsonl is stale; run scripts/build_rag_chunks.py")
    else:
        write_jsonl(chunks_path, chunks)
        write_jsonl(evalset_path, records)
    print(f"Validated {len(chunks)} chunks and {len(records)} eval records ({len(referenced)} references).")


if __name__ == "__main__":
    main()
