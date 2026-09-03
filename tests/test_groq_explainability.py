import numpy as np
import pandas as pd
import pytest

from credit_risk.explainability import CreditRiskExplainer, ExplanationContext
from credit_risk.pipeline import CreditRiskPipeline


FEATURES = ["f1", "f2", "f3", "f4", "f5", "f6", "f7"]
X = pd.DataFrame(
    [[10, 20, 30, 40, 50, 60, 70], [11, 21, 31, 41, 51, 61, 71]],
    columns=FEATURES,
)
SHAP_VALUES = np.array(
    [
        [0.1, -0.2, 0.7, -0.8, 0.3, -0.4, 0.05],
        [-1.0, 2.0, -3.0, 4.0, -5.0, 6.0, 0.0],
    ]
)


def make_explainer():
    return CreditRiskExplainer(
        SHAP_VALUES,
        FEATURES,
        api_key="test-key",
        model_name="XGBoost",
    )


def test_missing_api_key_is_safe(monkeypatch):
    monkeypatch.delenv("GROQ_API_KEY", raising=False)
    with pytest.raises(ValueError, match="GROQ_API_KEY not found"):
        CreditRiskExplainer(SHAP_VALUES, FEATURES)


@pytest.mark.parametrize(
    ("probability", "expected"),
    [(0.23998729, 0), (0.23998730, 1), (0.9, 1)],
)
def test_predicted_class_uses_decision_threshold(probability, expected):
    context = make_explainer()._build_context(0, X, probability)
    assert context.predicted_class == expected


def test_decision_threshold_and_risk_band_are_independent():
    context = make_explainer()._build_context(0, X, 0.25)
    assert context.predicted_class == 1
    assert context.risk_band == "BAJO"


def test_context_and_top_shap_are_aligned_to_same_row():
    context = make_explainer()._build_context(1, X, 0.5)
    assert isinstance(context, ExplanationContext)
    assert context.model_name == "XGBoost"
    assert context.probability_default == 0.5
    assert [(c.feature, c.value, c.shap_value) for c in context.positive_contributions] == [
        ("f6", 61, 6.0), ("f4", 41, 4.0), ("f2", 21, 2.0)
    ]
    assert [(c.feature, c.value, c.shap_value) for c in context.negative_contributions] == [
        ("f5", 51, -5.0), ("f3", 31, -3.0), ("f1", 11, -1.0)
    ]


def test_prompt_has_dynamic_prediction_metadata_and_safety_rules():
    explainer = make_explainer()
    prompt = explainer._build_prompt(explainer._build_context(0, X, 0.25))
    for text in (
        "XGBoost", "probability_default", "predicted_class",
        "decision_threshold", "risk_band", "No inventes información",
        "no deben interpretarse como relaciones causales",
    ):
        assert text in prompt


def test_http_is_mocked_and_structured_result_preserves_narrative(monkeypatch):
    class Response:
        def raise_for_status(self):
            pass

        def json(self):
            return {"choices": [{"message": {"content": "Narrativa breve"}}]}

    captured = {}

    def fake_post(url, **kwargs):
        captured.update(url=url, **kwargs)
        return Response()

    monkeypatch.setattr("credit_risk.explainability.requests.post", fake_post)
    result = make_explainer().explain_result(0, X, 0.25)
    assert result.narrative == "Narrativa breve"
    assert result.predicted_class == 1
    assert captured["json"]["model"] == "openai/gpt-oss-20b"
    assert "test-key" not in captured["json"]["messages"][0]["content"]


def test_pipeline_keeps_binary_decision_separate_from_risk_band():
    class Predictor:
        def predict_proba(self, frame):
            return np.array([0.25, 0.70])

    result = CreditRiskPipeline(Predictor()).evaluar_cliente(0, X)
    assert result["predicted_class"] == 1
    assert result["decision_threshold"] == 0.23998730
    assert result["nivel_riesgo"] == "BAJO"
