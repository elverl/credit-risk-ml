"""Small local dense retriever based on latent semantic analysis (LSA)."""

from __future__ import annotations

import hashlib
import json
import pickle
from collections import defaultdict
from pathlib import Path
from typing import Any, Iterable

import numpy as np
from sklearn.decomposition import TruncatedSVD
from sklearn.feature_extraction.text import TfidfVectorizer
from sklearn.preprocessing import normalize


EMBEDDING_MODEL = "local-lsa-tfidf-svd-v1"
DEFAULT_DIMENSION = 55
RANDOM_STATE = 42


def load_corpus(path: Path) -> list[dict[str, Any]]:
    """Load and validate the chunk corpus."""
    rows = [json.loads(line) for line in path.read_text(encoding="utf-8").splitlines() if line.strip()]
    if not rows:
        raise ValueError("Chunk corpus is empty")
    required = {"chunk_id", "source", "section_path", "text"}
    for index, row in enumerate(rows):
        missing = required - row.keys()
        if missing:
            raise ValueError(f"Chunk {index} is missing fields: {sorted(missing)}")
        if not str(row["text"]).strip():
            raise ValueError(f"Chunk {row['chunk_id']} has empty text")
    identifiers = [row["chunk_id"] for row in rows]
    if len(identifiers) != len(set(identifiers)):
        raise ValueError("Chunk IDs are not unique")
    return rows


def _retrieval_text(chunk: dict[str, Any]) -> str:
    metadata = " ".join(
        str(value)
        for value in (
            chunk["section_path"],
            chunk.get("feature_name", ""),
            chunk.get("target_name", ""),
            " ".join(chunk.get("tags", [])),
        )
        if value
    )
    return f"{metadata}\n{chunk['text']}"


def _corpus_sha256(corpus_path: Path) -> str:
    return hashlib.sha256(corpus_path.read_bytes()).hexdigest()


class DenseRetriever:
    """Cosine Top-K retrieval over deterministic dense LSA embeddings."""

    def __init__(
        self,
        chunks: list[dict[str, Any]],
        vectorizer: TfidfVectorizer,
        projector: TruncatedSVD,
        embeddings: np.ndarray,
    ) -> None:
        self.chunks = chunks
        self.vectorizer = vectorizer
        self.projector = projector
        self.embeddings = np.asarray(embeddings, dtype=np.float32)
        if self.embeddings.ndim != 2 or self.embeddings.shape[0] != len(chunks):
            raise ValueError("Embedding matrix does not match corpus")
        if not np.isfinite(self.embeddings).all():
            raise ValueError("Embedding matrix contains non-finite values")

    @classmethod
    def fit(cls, chunks: list[dict[str, Any]], dimension: int = DEFAULT_DIMENSION) -> "DenseRetriever":
        documents = [_retrieval_text(chunk) for chunk in chunks]
        vectorizer = TfidfVectorizer(
            lowercase=True,
            strip_accents="unicode",
            ngram_range=(1, 2),
            min_df=1,
            sublinear_tf=True,
            norm="l2",
        )
        sparse = vectorizer.fit_transform(documents)
        max_dimension = min(sparse.shape[0] - 1, sparse.shape[1] - 1)
        actual_dimension = min(dimension, max_dimension)
        if actual_dimension < 1:
            raise ValueError("Corpus is too small to build dense LSA embeddings")
        projector = TruncatedSVD(n_components=actual_dimension, random_state=RANDOM_STATE)
        embeddings = normalize(projector.fit_transform(sparse), norm="l2").astype(np.float32)
        return cls(chunks, vectorizer, projector, embeddings)

    @classmethod
    def load_or_build(
        cls,
        corpus_path: Path,
        artifact_dir: Path,
        dimension: int = DEFAULT_DIMENSION,
    ) -> "DenseRetriever":
        artifact_dir.mkdir(parents=True, exist_ok=True)
        model_path = artifact_dir / "dense_retriever.pkl"
        embeddings_path = artifact_dir / "kb_embeddings.npz"
        manifest_path = artifact_dir / "dense_retriever_config.json"
        corpus_hash = _corpus_sha256(corpus_path)
        expected = {
            "embedding_model": EMBEDDING_MODEL,
            "requested_dimension": dimension,
            "random_state": RANDOM_STATE,
            "corpus_sha256": corpus_hash,
        }
        if model_path.exists() and embeddings_path.exists() and manifest_path.exists():
            manifest = json.loads(manifest_path.read_text(encoding="utf-8"))
            if all(manifest.get(key) == value for key, value in expected.items()):
                with model_path.open("rb") as stream:
                    vectorizer, projector = pickle.load(stream)  # trusted local artifact
                matrix = np.load(embeddings_path)["embeddings"]
                return cls(load_corpus(corpus_path), vectorizer, projector, matrix)

        retriever = cls.fit(load_corpus(corpus_path), dimension=dimension)
        with model_path.open("wb") as stream:
            pickle.dump((retriever.vectorizer, retriever.projector), stream, protocol=pickle.HIGHEST_PROTOCOL)
        np.savez_compressed(embeddings_path, embeddings=retriever.embeddings)
        manifest = {
            **expected,
            "embedding_dimension": int(retriever.embeddings.shape[1]),
            "chunk_count": len(retriever.chunks),
            "similarity": "cosine (dot product over L2-normalized embeddings)",
            "vectorizer": {
                "type": "TfidfVectorizer",
                "ngram_range": [1, 2],
                "strip_accents": "unicode",
                "sublinear_tf": True,
            },
            "projector": "TruncatedSVD",
        }
        manifest_path.write_text(json.dumps(manifest, ensure_ascii=False, indent=2) + "\n", encoding="utf-8")
        return retriever

    def embed_query(self, query: str) -> np.ndarray:
        if not query or not query.strip():
            raise ValueError("Query must not be empty")
        sparse = self.vectorizer.transform([query])
        embedding = normalize(self.projector.transform(sparse), norm="l2").astype(np.float32)[0]
        if not np.isfinite(embedding).all():
            raise ValueError("Query embedding contains non-finite values")
        return embedding

    def search(self, query: str, top_k: int = 3) -> list[dict[str, Any]]:
        if top_k < 1:
            raise ValueError("top_k must be positive")
        query_embedding = self.embed_query(query)
        scores = self.embeddings @ query_embedding
        # Stable tie-breaking follows deterministic corpus order.
        ranked = np.argsort(-scores, kind="stable")[: min(top_k, len(self.chunks))]
        return [
            {
                "chunk_id": self.chunks[index]["chunk_id"],
                "score": float(scores[index]),
                "source": self.chunks[index]["source"],
                "section_path": self.chunks[index]["section_path"],
                "text": self.chunks[index]["text"],
            }
            for index in ranked
        ]


def evaluate_results(records: Iterable[dict[str, Any]]) -> dict[str, Any]:
    """Compute Hit Rate and reciprocal-rank aggregates at the recorded K."""
    rows = list(records)
    if not rows:
        raise ValueError("No retrieval results to evaluate")
    category_values: dict[str, list[tuple[float, float]]] = defaultdict(list)
    for row in rows:
        hit = float(bool(row["hit_at_3"]))
        rank = row.get("first_correct_rank")
        reciprocal_rank = 1.0 / rank if rank else 0.0
        category_values[row["category"]].append((hit, reciprocal_rank))
    by_category = {}
    for category, values in sorted(category_values.items()):
        by_category[category] = {
            "cases": len(values),
            "hits": int(sum(value[0] for value in values)),
            "hit_rate_at_3": sum(value[0] for value in values) / len(values),
            "mrr_at_3": sum(value[1] for value in values) / len(values),
        }
    reciprocal_ranks = [1.0 / row["first_correct_rank"] if row.get("first_correct_rank") else 0.0 for row in rows]
    hits = sum(bool(row["hit_at_3"]) for row in rows)
    return {
        "cases": len(rows),
        "hits": hits,
        "misses": len(rows) - hits,
        "hit_rate_at_3": hits / len(rows),
        "mrr_at_3": sum(reciprocal_ranks) / len(rows),
        "by_category": by_category,
    }
