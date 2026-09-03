"""Run exactly one authorized Groq baseline test on Test row 0."""

import os
from pathlib import Path

import pandas as pd

from credit_risk.explainability import CreditRiskExplainer, compute_shap_explanation
from credit_risk.pipeline import CreditRiskPipeline
from credit_risk.predictors import PicklePredictor


ROOT = Path(__file__).resolve().parents[1]
X_TEST_PATH = ROOT / "data" / "processed" / "X_test.csv"
MODEL_PATH = ROOT / "artifacts" / "models" / "xgboost.pkl"


def main() -> None:
    api_key = os.getenv("GROQ_API_KEY")
    if not api_key:
        raise SystemExit("GROQ_API_KEY is not configured; no remote call was made.")

    # The processed feature matrix contains no personal identifiers. Row 0 is
    # fixed so this command performs one reproducible, controlled request.
    X_test = pd.read_csv(X_TEST_PATH)
    predictor = PicklePredictor(MODEL_PATH)
    row = X_test.iloc[[0]]
    shap_result = compute_shap_explanation(predictor.model, row)
    explainer = CreditRiskExplainer(
        shap_values=shap_result["values"],
        features=shap_result["feature_names"],
        api_key=api_key,
        model_name="XGBoost",
    )
    result = CreditRiskPipeline(predictor, explainer).evaluar_cliente(0, row)
    print(result["explanation_details"])


if __name__ == "__main__":
    main()
