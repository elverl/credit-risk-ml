"""Deterministic Baseline-vs-RAG evaluation utilities."""

from __future__ import annotations

import json
import math
import re
import time
import unicodedata
from collections import defaultdict
from dataclasses import asdict, dataclass
from typing import Any

from credit_risk.explainability import GROQ_MODEL, GroqChatClient

from .pipeline import RAG_INSTRUCTIONS, enforce_grounding_guardrails


STOPWORDS = {
    "a", "al", "algo", "como", "con", "de", "del", "el", "en", "es", "esa",
    "ese", "esta", "este", "la", "las", "lo", "los", "más", "no", "o", "para",
    "por", "que", "se", "sin", "su", "sus", "un", "una", "y", "dentro", "cuando",
}
ABSTENTION_MARKERS = ("no documentado", "no existe una política", "evidencia insuficiente")
CAUSALITY_DENIALS = ("no demuestra causalidad", "no demuestran causalidad", "no prueba causalidad", "no son causas")
RECOMMENDATION_TERMS = ("debe aprobarse", "debe rechazarse", "recomiendo aprobar", "recomiendo rechazar")


@dataclass(frozen=True)
class BaselineResult:
    query: str
    model_name: str
    latency_seconds: float
    generated_answer: str
    usage: dict[str, Any] | None
    finish_reason: str | None
    model_context: dict[str, Any] | None

    def to_dict(self) -> dict[str, Any]:
        return asdict(self)


def build_baseline_prompt(query: str, model_context: dict[str, Any] | None = None) -> str:
    context = (
        json.dumps(model_context, ensure_ascii=False, indent=2)
        if model_context is not None else "No proporcionado para esta consulta."
    )
    return f"""# PREGUNTA
{query}

# CONTEXTO RECUPERADO
No proporcionado. Este es el baseline sin retrieval.

# CONTEXTO DEL MODELO Y SHAP
{context}

# INSTRUCCIONES
{RAG_INSTRUCTIONS}
- Como no hay chunks recuperados, no inventes ni listes chunk_id o fuentes.
"""


class BaselinePipeline:
    """Same Groq generation contract as RAG, deliberately without retrieval."""

    def __init__(self, generator=None, api_key: str | None = None):
        self.generator = generator or GroqChatClient(api_key=api_key, model=GROQ_MODEL)

    def answer(self, query: str, model_context: dict[str, Any] | None = None) -> BaselineResult:
        started = time.perf_counter()
        generation = self.generator.generate(build_baseline_prompt(query, model_context), max_tokens=1200)
        answer = enforce_grounding_guardrails(query, generation.content, model_context)
        return BaselineResult(
            query=query,
            model_name=generation.model,
            latency_seconds=time.perf_counter() - started,
            generated_answer=answer,
            usage=generation.usage,
            finish_reason=generation.finish_reason,
            model_context=model_context,
        )


def normalize_text(value: str) -> str:
    value = unicodedata.normalize("NFKD", value.lower())
    value = "".join(char for char in value if not unicodedata.combining(char))
    return re.sub(r"\s+", " ", re.sub(r"[^a-z0-9_.+%/-]+", " ", value)).strip()


def _tokens(value: str) -> set[str]:
    return {
        token for token in normalize_text(value).split()
        if token not in STOPWORDS and (len(token) > 2 or any(char.isdigit() for char in token))
    }


def extract_numbers(value: str) -> list[float]:
    values = []
    for raw, percent in re.findall(r"(?<![\w])(-?\d+(?:[.,]\d+)?)\s*(%)?", value):
        number = float(raw.replace(",", "."))
        values.append(number / 100.0 if percent else number)
    return values


def numbers_equivalent(expected: float, observed: float, tolerance: float = 5e-4) -> bool:
    candidates = (observed, observed / 100.0, observed * 100.0)
    return any(math.isclose(expected, candidate, rel_tol=tolerance, abs_tol=tolerance) for candidate in candidates)


def score_expected_fact(fact: str, answer: str) -> dict[str, Any]:
    expected_tokens = _tokens(fact)
    answer_tokens = _tokens(answer)
    concept_score = len(expected_tokens & answer_tokens) / len(expected_tokens) if expected_tokens else 1.0
    expected_numbers = extract_numbers(fact)
    observed_numbers = extract_numbers(answer)
    numeric_ok = all(any(numbers_equivalent(number, seen) for seen in observed_numbers) for number in expected_numbers)
    passed = numeric_ok and concept_score >= 0.45
    return {"fact": fact, "passed": passed, "concept_score": concept_score, "numeric_ok": numeric_ok}


def score_factual_correctness(expected_facts: list[str], answer: str) -> tuple[float, list[dict[str, Any]]]:
    details = [score_expected_fact(fact, answer) for fact in expected_facts]
    return sum(item["passed"] for item in details) / len(details), details


def score_groundedness(answer: str, context: str) -> float:
    """Average lexical/numeric support of answer sentences in allowed context."""
    clean_answer = re.sub(r"Fuentes recuperadas:[\s\S]*$", "", answer, flags=re.IGNORECASE)
    if any(marker in normalize_text(clean_answer) for marker in ABSTENTION_MARKERS):
        # An explicit refusal makes no positive project claim; correctness and
        # relevance separately penalize abstaining on an answerable question.
        return 1.0
    sentences = [part.strip() for part in re.split(r"[\n.!?]+", clean_answer) if len(_tokens(part)) >= 2]
    context_tokens = _tokens(context)
    context_numbers = extract_numbers(context)
    if not sentences:
        return 0.0
    scores = []
    for sentence in sentences:
        tokens = _tokens(sentence)
        lexical = len(tokens & context_tokens) / len(tokens) if tokens else 0.0
        numbers = extract_numbers(sentence)
        if numbers:
            numeric = sum(
                any(numbers_equivalent(number, seen) for seen in context_numbers) for number in numbers
            ) / len(numbers)
            scores.append(0.8 * lexical + 0.2 * numeric)
        else:
            scores.append(lexical)
    return sum(scores) / len(scores)


def score_answer_relevance(question: str, expected_facts: list[str], answer: str) -> float:
    if not answer.strip():
        return 0.0
    answer_tokens = _tokens(answer)
    question_tokens = _tokens(question)
    expected_tokens = set().union(*(_tokens(fact) for fact in expected_facts))
    question_overlap = len(question_tokens & answer_tokens) / len(question_tokens) if question_tokens else 1.0
    expected_overlap = len(expected_tokens & answer_tokens) / len(expected_tokens) if expected_tokens else 1.0
    return min(1.0, 0.5 * question_overlap + 0.5 * expected_overlap)


def score_abstention(expected_abstention: bool, answer: str) -> float:
    normalized = normalize_text(answer)
    abstained = any(marker in normalized for marker in ABSTENTION_MARKERS)
    unsafe = any(term in normalized for term in RECOMMENDATION_TERMS)
    return float((abstained and not unsafe) if expected_abstention else (not abstained and not unsafe))


def score_integrated_prediction(model_context: dict[str, Any], answer: str) -> tuple[float, dict[str, bool]]:
    normalized = normalize_text(answer)
    numbers = extract_numbers(answer)
    checks = {
        "probability_default": any(numbers_equivalent(model_context["probability_default"], value) for value in numbers),
        "decision_threshold": any(numbers_equivalent(model_context["decision_threshold"], value) for value in numbers),
        "predicted_class": bool(re.search(rf"(?:clase|predicted_class)[^\n.]{{0,35}}\b{model_context['predicted_class']}\b", normalized)),
        "risk_band": str(model_context["risk_band"]).lower() in normalized,
        "shap_directions": all(
            feature["feature"].lower() in normalized and feature["direction"].lower()[:3] in normalized
            for feature in model_context.get("top_shap_features", [])
        ),
        "non_causality": any(term in normalized for term in CAUSALITY_DENIALS),
        "no_credit_recommendation": not any(term in normalized for term in RECOMMENDATION_TERMS),
    }
    return sum(checks.values()) / len(checks), checks


def evaluate_answer(case: dict[str, Any], response: dict[str, Any], allowed_context: str) -> dict[str, Any]:
    answer = response["generated_answer"]
    factual, facts = score_factual_correctness(case["expected_facts"], answer)
    integrated, integrated_checks = (None, None)
    if case.get("model_context"):
        integrated, integrated_checks = score_integrated_prediction(case["model_context"], answer)
    return {
        "factual_correctness": factual,
        "fact_details": facts,
        "groundedness": score_groundedness(answer, allowed_context),
        "answer_relevance": score_answer_relevance(case["question"], case["expected_facts"], answer),
        "abstention_accuracy": score_abstention(bool(case["expected_abstention"]), answer),
        "integrated_prediction_consistency": integrated,
        "integrated_checks": integrated_checks,
    }


def aggregate_comparison(rows: list[dict[str, Any]]) -> dict[str, Any]:
    quality = ("factual_correctness", "groundedness", "answer_relevance", "abstention_accuracy")
    systems = ("baseline", "rag")

    def aggregate(group: list[dict[str, Any]]) -> dict[str, Any]:
        output = {}
        for system in systems:
            metrics = {key: sum(row[system][key] for row in group) / len(group) for key in quality}
            integrated = [row[system]["integrated_prediction_consistency"] for row in group
                          if row[system]["integrated_prediction_consistency"] is not None]
            metrics["integrated_prediction_consistency"] = sum(integrated) / len(integrated) if integrated else None
            output[system] = metrics
        output["delta_rag_minus_baseline"] = {
            key: (output["rag"][key] - output["baseline"][key]
                  if output["rag"][key] is not None else None)
            for key in (*quality, "integrated_prediction_consistency")
        }
        return output

    categories: dict[str, list[dict[str, Any]]] = defaultdict(list)
    for row in rows:
        categories[row["category"]].append(row)
    return {
        "cases": len(rows),
        "global": aggregate(rows),
        "by_category": {category: {"cases": len(group), **aggregate(group)}
                        for category, group in sorted(categories.items())},
    }
