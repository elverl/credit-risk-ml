"""Explainability components for credit-risk predictions."""

import os
from abc import ABC, abstractmethod
from dataclasses import asdict, dataclass

import numpy as np
import pandas as pd
import requests
import shap
from dotenv import load_dotenv
from lightgbm import LGBMClassifier
from sklearn.ensemble import RandomForestClassifier
from sklearn.linear_model import LogisticRegression
from sklearn.pipeline import Pipeline
from sklearn.utils.validation import check_is_fitted
from xgboost import XGBClassifier


load_dotenv()

GROQ_API_URL = "https://api.groq.com/openai/v1/chat/completions"
GROQ_MODEL = "openai/gpt-oss-20b"


@dataclass(frozen=True)
class GroqGeneration:
    """Normalized response from the shared Groq chat-completions transport."""

    content: str
    model: str
    usage: dict | None
    finish_reason: str | None = None


class GroqChatClient:
    """Small shared Groq transport used by baseline and RAG generation."""

    def __init__(self, api_key: str | None = None, model: str = GROQ_MODEL):
        self.api_key = api_key or os.getenv("GROQ_API_KEY")
        self.model = model
        self.api_url = GROQ_API_URL
        if not self.api_key:
            raise ValueError(
                "GROQ_API_KEY not found. Configure it securely before "
                "requesting a Groq explanation."
            )

    def generate(self, prompt: str, max_tokens: int = 500) -> GroqGeneration:
        response = requests.post(
            self.api_url,
            headers={
                "Authorization": f"Bearer {self.api_key}",
                "Content-Type": "application/json",
            },
            json={
                "model": self.model,
                "messages": [{"role": "user", "content": prompt}],
                "max_tokens": max_tokens,
            },
            timeout=30,
        )
        response.raise_for_status()
        payload = response.json()
        return GroqGeneration(
            content=payload["choices"][0]["message"]["content"],
            model=str(payload.get("model", self.model)),
            usage=payload.get("usage"),
            finish_reason=payload["choices"][0].get("finish_reason"),
        )


@dataclass(frozen=True)
class ShapContribution:
    """One local model contribution, aligned with its observed value."""

    feature: str
    description: str
    value: object
    shap_value: float


@dataclass(frozen=True)
class ExplanationContext:
    """Minimal, non-identifying evidence sent to the LLM.

    ``decision_threshold`` determines class 0/1. ``risk_band`` is a
    descriptive probability segmentation and does not determine the class.
    """

    model_name: str
    probability_default: float
    predicted_class: int
    decision_threshold: float
    risk_band: str
    positive_contributions: tuple[ShapContribution, ...]
    negative_contributions: tuple[ShapContribution, ...]


@dataclass(frozen=True)
class ExplanationResult:
    """Structured LLM result; ``narrative`` preserves the legacy string."""

    narrative: str
    model_name: str
    probability_default: float
    predicted_class: int
    decision_threshold: float
    risk_band: str

    def to_dict(self) -> dict:
        return asdict(self)


DICCIONARIO_VARIABLES = {
    "n_atr_1d_6":
        "Número de atrasos mayores a 1 día en los últimos 6 meses",

    "cl_9_1.0":
        "Calificación CPP en los últimos 9 meses",

    "cl_12_1.0":
        "Calificación CPP en los últimos 12 meses",

    "cl_18_2.0":
        "Calificación Deficiente en los últimos 18 meses",

    "cl_9_3.0":
        "Calificación Dudoso en los últimos 9 meses",

    "ind_vjrc_36_1":
        "Tuvo crédito vencido/refinanciado/castigado en últimos 36 meses",

    "ind_per_x9_1.0":
        "Deuda bancaria personal consecutiva en últimos 9 meses",

    "prc_u_tc_cns":
        "Máximo porcentaje de utilización de tarjetas de crédito de consumo",

    "tas_u_tc":
        "Porcentaje de utilización de tarjetas de crédito en el último mes",

    "c_tc_mn_24":
        "Cantidad mínima de tarjetas de crédito en últimos 24 meses",

    "c_tc_mn_24_sld":
        "Cantidad mínima de tarjetas con saldo mayor a 0 en últimos 24 meses",

    "c_pcns":
        "Número de créditos de consumo al momento de evaluación",

    "vr_s_t_36":
        "Variación porcentual del saldo de deuda respecto a 36 meses atrás",

    "mx_ltc_36":
        "Máximo monto de línea de tarjeta de crédito en últimos 36 meses",

    "ipc_t6":
        "Tasa de inflación anual de hace 6 meses",
}


class BaseExplainer(ABC):
    """Base interface for credit-risk explainers."""

    @abstractmethod
    def explain(
        self,
        cliente_idx: int,
        X: pd.DataFrame,
        prob: float,
    ) -> str:
        """Generate a natural-language explanation."""
        raise NotImplementedError


def compute_shap_explanation(
    model,
    X: pd.DataFrame,
    background: pd.DataFrame | None = None,
) -> dict:
    """Compute a normalized SHAP explanation for a supported V2 model.

    Tree models use ``shap.TreeExplainer``. Logistic Regression uses
    ``shap.LinearExplainer`` on the feature space produced by its pipeline.
    The returned ``values`` always represent the positive class and have
    shape ``(n_observations, n_features)``.
    """
    if not isinstance(X, pd.DataFrame):
        raise TypeError("X must be a pandas DataFrame.")

    if background is not None:
        if not isinstance(background, pd.DataFrame):
            raise TypeError("background must be a pandas DataFrame.")
        if list(background.columns) != list(X.columns):
            raise ValueError(
                "X and background must have identical columns and order."
            )

    check_is_fitted(model)

    feature_names = list(X.columns)
    model_feature_names = getattr(model, "feature_names_in_", None)
    if (
        model_feature_names is not None
        and list(model_feature_names) != feature_names
    ):
        raise ValueError(
            "X columns and order do not match the fitted model."
        )

    classes = np.asarray(getattr(model, "classes_", []))
    positive_positions = np.flatnonzero(classes == 1)
    if len(positive_positions) != 1:
        raise ValueError(
            "The fitted model must contain exactly one positive class 1."
        )
    positive_class_index = int(positive_positions[0])

    if isinstance(model, Pipeline):
        estimator = model.named_steps.get("model")
        if not isinstance(estimator, LogisticRegression):
            raise TypeError(
                "Only LogisticRegression pipelines are supported by "
                "the linear SHAP integration."
            )
        if positive_class_index != 1:
            raise ValueError(
                "Binary Logistic Regression must use class 1 as the "
                "second class for LinearExplainer."
            )

        transformer = model[:-1]
        transformed_X = transformer.transform(X)
        background_X = X if background is None else background
        if len(background_X) > 1000:
            background_X = background_X.sample(
                n=1000,
                random_state=42,
            )
        transformed_background = transformer.transform(background_X)
        explainer = shap.LinearExplainer(
            estimator,
            transformed_background,
        )
        explanation = explainer(transformed_X)
        values = np.asarray(explanation.values)
        base_values = np.asarray(explanation.base_values)
        explainer_type = "LinearExplainer"

    elif isinstance(
        model,
        (RandomForestClassifier, XGBClassifier, LGBMClassifier),
    ):
        explainer = shap.TreeExplainer(model)
        explanation = explainer(X)
        values = np.asarray(explanation.values)
        base_values = np.asarray(explanation.base_values)

        if values.ndim == 3:
            values = values[:, :, positive_class_index]
            if base_values.ndim == 2:
                base_values = base_values[:, positive_class_index]

        explainer_type = "TreeExplainer"

    else:
        raise TypeError(
            "Unsupported model for SHAP explanation: "
            f"{type(model).__name__}."
        )

    if values.ndim != 2 or values.shape[1] != len(feature_names):
        raise ValueError(
            "Unexpected SHAP values shape: "
            f"{values.shape}; expected (*, {len(feature_names)})."
        )

    return {
        "values": values,
        "base_values": base_values,
        "data": X.to_numpy(),
        "feature_names": feature_names,
        "explainer_type": explainer_type,
        "positive_class": 1,
    }


class CreditRiskExplainer(BaseExplainer):
    """
    Generate natural-language explanations using SHAP and a Groq LLM.
    """

    def __init__(
        self,
        shap_values: np.ndarray,
        features: list[str],
        api_key: str | None = None,
        model: str = GROQ_MODEL,
        model_name: str = "modelo de riesgo crediticio",
    ):
        self.shap_values = shap_values
        self.features = features
        self.model_name = model_name
        self.client = GroqChatClient(api_key=api_key, model=model)
        self.api_key = self.client.api_key
        self.model = self.client.model
        self.api_url = self.client.api_url

    def _build_context(
        self,
        cliente_idx: int,
        X: pd.DataFrame,
        prob: float,
        decision_threshold: float = 0.23998730,
        risk_band: str | None = None,
    ) -> ExplanationContext:
        """Build minimal evidence for one row and positive class ``1``."""

        if not 0 <= cliente_idx < len(X):
            raise IndexError("cliente_idx is outside X.")
        if self.shap_values.ndim != 2:
            raise ValueError("shap_values must be a 2D array.")
        if self.shap_values.shape != (len(X), len(self.features)):
            raise ValueError("shap_values, X rows, and features are not aligned.")
        if list(X.columns) != list(self.features):
            raise ValueError("X columns and features must have identical order.")
        if not 0 <= decision_threshold <= 1:
            raise ValueError("decision_threshold must be between 0 and 1.")

        shap_values_client = self.shap_values[cliente_idx]
        values_client = X.iloc[cliente_idx]

        shap_df = pd.DataFrame(
            {
                "feature": self.features,
                "shap": shap_values_client,
                "valor": values_client.values,
                "descripcion": [
                    DICCIONARIO_VARIABLES.get(feature, feature)
                    for feature in self.features
                ],
            }
        )

        positive = shap_df[shap_df["shap"] > 0].nlargest(3, "shap")
        negative = shap_df[shap_df["shap"] < 0].nsmallest(3, "shap")

        def records(frame: pd.DataFrame) -> tuple[ShapContribution, ...]:
            return tuple(
                ShapContribution(
                    feature=str(row.feature),
                    description=str(row.descripcion),
                    value=row.valor,
                    shap_value=float(row.shap),
                )
                for row in frame.itertuples(index=False)
            )

        if risk_band is None:
            risk_band = "ALTO" if prob >= 0.60 else "MEDIO" if prob >= 0.30 else "BAJO"

        return ExplanationContext(
            model_name=self.model_name,
            probability_default=float(prob),
            predicted_class=int(prob >= decision_threshold),
            decision_threshold=float(decision_threshold),
            risk_band=risk_band,
            positive_contributions=records(positive),
            negative_contributions=records(negative),
        )

    @staticmethod
    def _format_contributions(items: tuple[ShapContribution, ...]) -> str:
        if not items:
            return "No hay contribuciones con este signo entre las suministradas."
        return "\n".join(
            f"- {item.description} (feature={item.feature}, valor={item.value}, "
            f"SHAP={item.shap_value:.6g})"
            for item in items
        )

    def _build_prompt(self, context: ExplanationContext) -> str:
        """Render the structured evidence as a constrained Spanish prompt."""
        return f"""
Eres un analista de riesgo crediticio experto del sistema financiero peruano.

Resultado del modelo:
- model_name: {context.model_name}
- probability_default: {context.probability_default:.6f}
- predicted_class: {context.predicted_class}
- decision_threshold: {context.decision_threshold:.8f}
- risk_band: {context.risk_band}

El decision_threshold determina exclusivamente la clase 0/1. La risk_band es
una segmentación descriptiva de la probabilidad con límites 0.30 y 0.60; no
determina la clase y no debe confundirse con el decision_threshold.

Principales contribuciones locales que elevan el output del modelo:

{self._format_contributions(context.positive_contributions)}

Principales contribuciones locales que reducen el output del modelo:

{self._format_contributions(context.negative_contributions)}

Redacta como máximo 3 párrafos cortos, en español claro para un analista de
riesgo: (1) resultado del modelo; (2) contribuciones que elevan su output; y
(3) contribuciones que lo reducen.

Limítate estrictamente a la información suministrada. No inventes información
ni conviertas asociaciones del modelo en hechos sobre el cliente. Si la
evidencia no permite responder algo, indícalo explícitamente. No atribuyas
causalidad. Las contribuciones SHAP explican el comportamiento del modelo y no deben interpretarse como relaciones causales.
"""

    def explain_result(
        self,
        cliente_idx: int,
        X: pd.DataFrame,
        prob: float,
        decision_threshold: float = 0.23998730,
        risk_band: str | None = None,
    ) -> ExplanationResult:
        """Request one narrative and return it with prediction metadata."""
        context = self._build_context(
            cliente_idx, X, prob, decision_threshold, risk_band
        )
        prompt = self._build_prompt(context)

        narrative = self.client.generate(prompt, max_tokens=500).content
        return ExplanationResult(
            narrative=narrative,
            model_name=context.model_name,
            probability_default=context.probability_default,
            predicted_class=context.predicted_class,
            decision_threshold=context.decision_threshold,
            risk_band=context.risk_band,
        )

    def explain(
        self,
        cliente_idx: int,
        X: pd.DataFrame,
        prob: float,
    ) -> str:
        """Generate the credit-risk explanation."""

        return self.explain_result(cliente_idx, X, prob).narrative
