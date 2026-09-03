"""Contract tests for deterministic knowledge-base chunking."""

import json
from pathlib import Path

from credit_risk.config import FEATURES
from credit_risk.rag.chunking import build_chunks


ROOT = Path(__file__).resolve().parents[1]
KB_DIR = ROOT / "docs" / "knowledge_base"


def test_chunk_ids_are_unique_and_build_is_deterministic():
    first = build_chunks(KB_DIR)
    second = build_chunks(KB_DIR)
    assert first == second
    identifiers = [chunk["chunk_id"] for chunk in first]
    assert len(identifiers) == len(set(identifiers))
    assert all(chunk["text"].strip() for chunk in first)


def test_required_metadata_is_preserved():
    chunks = build_chunks(KB_DIR)
    assert {chunk["source"] for chunk in chunks} == {path.name for path in KB_DIR.glob("*.md")}
    for chunk in chunks:
        assert chunk["kb_version"] == 1
        assert chunk["model_version"] == "v2"
        assert chunk["status"] == "current"
        assert chunk["source"].endswith(".md")
        assert chunk["section_path"]
        assert chunk["doc_type"]
        assert isinstance(chunk["tags"], list) and chunk["tags"]


def test_all_contract_features_and_target_have_dedicated_chunks():
    chunks = build_chunks(KB_DIR)
    features = [chunk.get("feature_name") for chunk in chunks if "feature_name" in chunk]
    targets = [chunk.get("target_name") for chunk in chunks if "target_name" in chunk]
    assert features == FEATURES
    assert targets == ["incumplimiento"]


def test_evalset_expected_chunk_ids_resolve():
    chunks = build_chunks(KB_DIR)
    known_ids = {chunk["chunk_id"] for chunk in chunks}
    records = [json.loads(line) for line in (ROOT / "evaluation" / "rag_evalset_v1.jsonl").read_text(encoding="utf-8").splitlines()]
    assert all(record.get("expected_chunk_ids") for record in records)
    assert all(set(record["expected_chunk_ids"]) <= known_ids for record in records)
    by_id = {chunk["chunk_id"]: chunk for chunk in chunks}
    for record in records:
        for identifier in record["expected_chunk_ids"]:
            assert by_id[identifier]["source"] in record["expected_sources"]
            assert any(
                f" > {str(section).strip('`')}" in by_id[identifier]["section_path"]
                for section in record["expected_sections"]
            )


def test_corpus_excludes_v1_and_obsolete_llama_content():
    chunks = build_chunks(KB_DIR)
    corpus = "\n".join(str(chunk["text"]) for chunk in chunks).lower()
    assert "model_version: v1" not in corpus
    assert "llama-3.1-8b-instant" not in corpus
    assert "openai/gpt-oss-20b" in corpus
