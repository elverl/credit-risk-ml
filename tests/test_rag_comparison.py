"""Tests for deterministic Baseline-vs-RAG scorers."""

import pytest

from credit_risk.explainability import GroqGeneration
from credit_risk.rag.comparison import (
    BaselinePipeline,
    numbers_equivalent,
    score_abstention,
    score_expected_fact,
    score_factual_correctness,
    score_groundedness,
    score_integrated_prediction,
)


def test_expected_fact_uses_concepts_not_exact_string():
    score, details = score_factual_correctness(
        ["La ventana de performance es 12 meses desde la originación"],
        "Desde que se origina el crédito, la ventana de performance dura 12 meses.",
    )
    assert score == 1.0
    assert details[0]["passed"]


def test_numeric_equivalence_accepts_probability_and_percentage():
    assert numbers_equivalent(0.5732861757278442, 57.33)
    assert numbers_equivalent(0.2399872988462448, 0.23999)
    assert not numbers_equivalent(0.5732861757278442, 25.0)


def test_abstention_rejects_prescriptive_answer():
    assert score_abstention(True, "No documentado en el proyecto.") == 1.0
    assert score_abstention(True, "Debe rechazarse el crédito.") == 0.0


def test_groundedness_treats_safe_abstention_separately_from_relevance():
    assert score_groundedness("No documentado en el proyecto.", "") == 1.0
    assert score_groundedness("Afirmación sin soporte.", "") == 0.0


def test_integrated_threshold_band_shap_and_non_causality():
    context = {
        "probability_default": 0.25935208797454834,
        "predicted_class": 1,
        "decision_threshold": 0.2399872988462448,
        "risk_band": "BAJO",
        "top_shap_features": [
            {"feature": "x", "direction": "positive"},
            {"feature": "y", "direction": "negative"},
        ],
    }
    answer = (
        "probability_default 25.9352%, sobre decision_threshold 0.2399873: clase 1. "
        "La banda es BAJO. x tiene dirección positive e y negative. "
        "SHAP no demuestra causalidad."
    )
    score, checks = score_integrated_prediction(context, answer)
    assert score == 1.0
    assert all(checks.values())


def test_baseline_groq_is_mocked_and_receives_no_chunks():
    class FakeGenerator:
        model = "openai/gpt-oss-20b"

        def __init__(self):
            self.prompt = ""

        def generate(self, prompt, max_tokens=1200):
            self.prompt = prompt
            return GroqGeneration("No documentado", self.model, {"total_tokens": 3}, "stop")

    generator = FakeGenerator()
    result = BaselinePipeline(generator=generator).answer("Pregunta")
    assert "baseline sin retrieval" in generator.prompt
    assert "[CHUNK" not in generator.prompt
    assert result.model_name == "openai/gpt-oss-20b"
