"""End-to-end RAG V1 pipeline: dense retrieval, grounded prompt, Groq."""

from __future__ import annotations

import json
import re
import time
from dataclasses import asdict, dataclass
from typing import Any, Protocol

from credit_risk.explainability import GROQ_MODEL, GroqChatClient, GroqGeneration

from .retriever import DenseRetriever


RAG_INSTRUCTIONS = """- Responde en español claro y conciso usando únicamente la evidencia proporcionada cuando la pregunta sea sobre conocimiento del proyecto.
- No inventes definiciones, políticas, reglas, hechos ni referencias. Asocia la respuesta solo con los chunk_id suministrados.
- Si la evidencia no permite responder, indica explícitamente: "No documentado en el proyecto" o que la evidencia es insuficiente.
- No inventes límites de crédito, pricing o tasas, políticas de aprobación/rechazo ni excepciones crediticias.
- Distingue probability_default, predicted_class, decision_threshold y risk_band. El threshold determina la clase; la banda es descriptiva y usa 0.30/0.60.
- Interpreta SHAP únicamente como contribución al output del modelo, nunca como causalidad, protección causal o hecho independiente sobre el cliente.
- No conviertas SHAP, una clase, una probabilidad o una banda en recomendación crediticia.
- Ante una pregunta causal sobre SHAP, corrige la premisa explicando que SHAP describe contribuciones al output; no te abstengas sin dar esa corrección.
- Para contexto de predicción, conserva exactamente los valores numéricos suministrados y explica sus comparaciones sin recalcularlos de forma contradictoria.
- Cierra con una línea "Fuentes: ..." que liste únicamente los chunk_id efectivamente usados. No cites chunks que no sustentan la respuesta."""


class Generator(Protocol):
    model: str

    def generate(self, prompt: str, max_tokens: int = 500) -> GroqGeneration: ...


@dataclass(frozen=True)
class RAGResult:
    query: str
    retrieved_chunk_ids: list[str]
    retrieval_scores: list[float]
    sources: list[dict[str, str]]
    model_name: str
    latency_seconds: float
    generated_answer: str
    usage: dict[str, Any] | None
    finish_reason: str | None
    model_context: dict[str, Any] | None

    def to_dict(self) -> dict[str, Any]:
        return asdict(self)


def build_knowledge_context(retrieved: list[dict[str, Any]]) -> str:
    """Render retrieved evidence with non-invented chunk references."""
    blocks = []
    for rank, chunk in enumerate(retrieved, start=1):
        blocks.append(
            f"[CHUNK {rank}]\n"
            f"chunk_id: {chunk['chunk_id']}\n"
            f"source: {chunk['source']}\n"
            f"section_path: {chunk['section_path']}\n"
            f"text:\n{chunk['text']}"
        )
    return "\n\n".join(blocks)


def build_rag_prompt(
    query: str,
    retrieved: list[dict[str, Any]],
    model_context: dict[str, Any] | None = None,
) -> str:
    """Build the explicit four-part grounded prompt contract."""
    prediction_context = (
        json.dumps(model_context, ensure_ascii=False, indent=2)
        if model_context is not None
        else "No proporcionado para esta consulta."
    )
    return f"""# PREGUNTA
{query}

# CONTEXTO RECUPERADO TOP-3
{build_knowledge_context(retrieved)}

# CONTEXTO DEL MODELO Y SHAP
{prediction_context}

# INSTRUCCIONES
{RAG_INSTRUCTIONS}
"""


def ground_answer_sources(answer: str, retrieved_ids: list[str]) -> str:
    """Replace model-authored reference footers with verified retrieved IDs."""
    clean = re.sub(r"\n+\s*(?:\*\*)?Fuentes(?: recuperadas)?(?:\*\*)?:[\s\S]*$", "", answer,
                   flags=re.IGNORECASE).rstrip()
    return clean + "\n\nFuentes recuperadas: " + ", ".join(
        f"`{identifier}`" for identifier in retrieved_ids
    )


def enforce_grounding_guardrails(query: str, answer: str, model_context: dict[str, Any] | None) -> str:
    """Enforce critical policy abstention and SHAP safety invariants."""
    normalized = query.lower()
    answer = re.sub(
        r"\n+\s*(?:\*\*)?Fuentes(?: recuperadas)?(?:\*\*)?:[\s\S]*$", "", answer,
        flags=re.IGNORECASE,
    ).rstrip()
    policy_rules = (
        (("límite de crédito", "limite de credito"),
         "No documentado en el proyecto. La knowledge base no contiene una política institucional de límites de crédito; la probabilidad no define un monto."),
        (("aprobar", "rechazar", "aprobación", "aprobacion"),
         "No documentado en el proyecto. No existe una política institucional de aprobación o rechazo; predicted_class=1 no equivale a rechazo automático."),
        (("pricing", "tasa de interés", "tasa de interes"),
         "No documentado en el proyecto. La banda es descriptiva y no existe una política institucional de pricing o tasas."),
        (("excepción crediticia", "excepcion crediticia"),
         "No documentado en el proyecto. No existe una política institucional de excepciones crediticias y la predicción no determina una excepción."),
    )
    for terms, safe_answer in policy_rules:
        if any(term in normalized for term in terms):
            return safe_answer

    if "shap" in normalized and any(term in normalized for term in ("causa", "causó", "causo")):
        return (
            "No puede identificarse una causa real con esta evidencia. SHAP describe "
            "contribuciones al output del modelo para una observación y no demuestra causalidad."
        )

    # Remove a known unsupported reconciliation claim if a model emits it.
    answer = re.sub(
        r"(?im)^.*(?:contribuciones|valores)\s+(?:SHAP\s+)?suman.*(?:valor base|probabilidad).*$",
        "",
        answer,
    ).strip()
    shap_related = "shap" in normalized or bool(model_context and model_context.get("top_shap_features"))
    if shap_related:
        disclaimer = (
            "Las contribuciones SHAP describen cambios en el output del modelo; "
            "no demuestran causalidad ni constituyen una recomendación crediticia."
        )
        if "no demuestran causalidad" not in answer.lower():
            answer = f"{answer}\n\n{disclaimer}".strip()
    return answer


class RAGPipeline:
    """Retrieve Top-3 evidence and request one grounded answer."""

    def __init__(
        self,
        retriever: DenseRetriever,
        generator: Generator | None = None,
        api_key: str | None = None,
    ) -> None:
        self.retriever = retriever
        self.generator = generator or GroqChatClient(api_key=api_key, model=GROQ_MODEL)

    def answer(self, query: str, model_context: dict[str, Any] | None = None) -> RAGResult:
        started = time.perf_counter()
        retrieved = self.retriever.search(query, top_k=3)
        prompt = build_rag_prompt(query, retrieved, model_context)
        generation = self.generator.generate(prompt, max_tokens=1200)
        # References are system-controlled: discard any malformed model footer
        # and attach only IDs that were truly supplied by retrieval.
        retrieved_ids = [item["chunk_id"] for item in retrieved]
        guarded = enforce_grounding_guardrails(query, generation.content, model_context)
        answer = ground_answer_sources(guarded, retrieved_ids)
        latency = time.perf_counter() - started
        return RAGResult(
            query=query,
            retrieved_chunk_ids=retrieved_ids,
            retrieval_scores=[item["score"] for item in retrieved],
            sources=[
                {
                    "chunk_id": item["chunk_id"],
                    "source": item["source"],
                    "section_path": item["section_path"],
                }
                for item in retrieved
            ],
            model_name=generation.model,
            latency_seconds=latency,
            generated_answer=answer,
            usage=generation.usage,
            finish_reason=generation.finish_reason,
            model_context=model_context,
        )
