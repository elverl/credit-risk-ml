"""High-level credit-risk prediction pipeline."""

import random

import pandas as pd

from .explainability import BaseExplainer
from .predictors import BasePredictor


class CreditRiskPipeline:
    """
    Pipeline for credit-risk prediction and explanation.

    The predictor and explainer are injected as dependencies.
    Risk-band thresholds are configurable.
    """

    def __init__(
        self,
        predictor: BasePredictor,
        explainer: BaseExplainer | None = None,
        low_threshold: float = 0.30,
        high_threshold: float = 0.60,
        decision_threshold: float = 0.23998730,
    ):
        if not 0 <= low_threshold < high_threshold <= 1:
            raise ValueError(
                "Thresholds must satisfy "
                "0 <= low_threshold < high_threshold <= 1."
            )
        if not 0 <= decision_threshold <= 1:
            raise ValueError("decision_threshold must be between 0 and 1.")

        self.predictor = predictor
        self.explainer = explainer

        self.low_threshold = low_threshold
        self.high_threshold = high_threshold
        self.decision_threshold = decision_threshold

    def classify_risk(self, probability: float) -> str:
        """
        Convert probability of default into a risk band.
        """

        if probability >= self.high_threshold:
            return "ALTO"

        if probability >= self.low_threshold:
            return "MEDIO"

        return "BAJO"

    def evaluar_cliente(
        self,
        cliente_idx: int,
        X: pd.DataFrame,
        y: pd.Series | None = None,
    ) -> dict:
        """
        Evaluate one customer.
        """

        probabilities = self.predictor.predict_proba(X)

        probability = float(
            probabilities[cliente_idx]
        )

        risk_band = self.classify_risk(probability)
        predicted_class = int(probability >= self.decision_threshold)
        result = {
            "cliente_idx": cliente_idx,
            "probabilidad": round(probability, 4),
            "nivel_riesgo": risk_band,
            "predicted_class": predicted_class,
            "decision_threshold": self.decision_threshold,
            "real": (
                int(y.iloc[cliente_idx])
                if y is not None
                else None
            ),
            "explicacion": None,
        }

        if self.explainer is not None:
            if hasattr(self.explainer, "explain_result"):
                explanation = self.explainer.explain_result(
                    cliente_idx, X, probability,
                    self.decision_threshold, risk_band,
                )
                result["explicacion"] = explanation.narrative
                result["explanation_details"] = explanation.to_dict()
            else:
                result["explicacion"] = self.explainer.explain(
                    cliente_idx, X, probability
                )

        return result

    def evaluar_aleatorio(
        self,
        X: pd.DataFrame,
        y: pd.Series | None = None,
    ) -> dict:
        """
        Evaluate a randomly selected customer.
        """

        cliente_idx = random.randint(
            0,
            len(X) - 1,
        )

        return self.evaluar_cliente(
            cliente_idx,
            X,
            y,
        )
