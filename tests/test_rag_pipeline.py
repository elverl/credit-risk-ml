"""Unit tests for the grounded RAG prompt and trace contract (no real API)."""

from types import SimpleNamespace

import pytest

from credit_risk.explainability import GroqGeneration
from credit_risk.rag.pipeline import (
    RAGPipeline,
    build_knowledge_context,
    build_rag_prompt,
    enforce_grounding_guardrails,
)


TOP_3 = [
    {"chunk_id": f"chunk-{i}", "score": 1.0 / i, "source": f"source-{i}.md",
     "section_path": f"Doc > Section {i}", "text": f"Evidence {i}"}
    for i in range(1, 4)
]


class FakeRetriever:
    def search(self, query, top_k=3):
        assert top_k == 3
        return TOP_3


class FakeGenerator:
    model = "openai/gpt-oss-20b"

    def __init__(self):
        self.prompt = None

    def generate(self, prompt, max_tokens=500):
        assert max_tokens == 1200
        self.prompt = prompt
        return GroqGeneration("Grounded answer", self.model, {"total_tokens": 42}, "stop")


def test_context_contains_top_3_with_traceable_sources():
    context = build_knowledge_context(TOP_3)
    assert all(item["chunk_id"] in context for item in TOP_3)
    assert all(item["source"] in context for item in TOP_3)
    assert context.count("[CHUNK ") == 3


def test_prompt_contract_has_sections_and_safety_instructions():
    prompt = build_rag_prompt("Question", TOP_3)
    for text in (
        "# PREGUNTA", "# CONTEXTO RECUPERADO TOP-3",
        "# CONTEXTO DEL MODELO Y SHAP", "# INSTRUCCIONES",
        "No documentado en el proyecto", "límites de crédito", "pricing",
        "aprobación/rechazo", "excepciones crediticias",
        "nunca como causalidad", "No conviertas SHAP",
    ):
        assert text in prompt


def test_pipeline_metadata_and_top_3_are_preserved():
    generator = FakeGenerator()
    context = {"probability_default": 0.25, "predicted_class": 1,
               "decision_threshold": 0.2399873, "risk_band": "BAJO",
               "top_shap_features": []}
    result = RAGPipeline(FakeRetriever(), generator=generator).answer("Question", context)
    assert result.retrieved_chunk_ids == ["chunk-1", "chunk-2", "chunk-3"]
    assert result.retrieval_scores == [1.0, 0.5, pytest.approx(1 / 3)]
    assert result.model_name == "openai/gpt-oss-20b"
    assert result.generated_answer.startswith("Grounded answer")
    assert "Fuentes recuperadas: `chunk-1`, `chunk-2`, `chunk-3`" in result.generated_answer
    assert result.usage == {"total_tokens": 42}
    assert result.finish_reason == "stop"
    assert result.model_context == context
    assert result.latency_seconds >= 0
    assert "0.2399873" in generator.prompt


def test_threshold_and_band_contract_are_explicit_for_ip_002():
    context = {"probability_default": 0.25935208797454834, "predicted_class": 1,
               "decision_threshold": 0.2399873, "risk_band": "BAJO",
               "top_shap_features": []}
    prompt = build_rag_prompt("Explain", TOP_3, context)
    assert '"predicted_class": 1' in prompt
    assert '"risk_band": "BAJO"' in prompt
    assert "El threshold determina la clase; la banda es descriptiva" in prompt


@pytest.mark.parametrize("query", [
    "¿Qué límite de crédito exacto corresponde?",
    "¿Debe aprobarse o rechazarse automáticamente?",
    "¿Qué pricing o tasa de interés corresponde?",
    "¿Qué excepción crediticia debe concederse?",
])
def test_policy_guardrail_forces_documented_abstention(query):
    answer = enforce_grounding_guardrails(query, "Debe aprobarse.", None)
    assert answer.startswith("No documentado en el proyecto")


def test_shap_guardrail_removes_reconciliation_and_adds_non_causality():
    answer = enforce_grounding_guardrails(
        "Explica SHAP",
        "Las contribuciones SHAP suman la diferencia entre el valor base y la probabilidad.\nOtro hecho.",
        {"top_shap_features": [{"feature": "x"}]},
    )
    assert "suman la diferencia" not in answer
    assert "no demuestran causalidad" in answer
