"""Inference checks for persisted predictors and feature contracts."""

import pickle

import pandas as pd
import pytest
from sklearn.linear_model import LogisticRegression

from credit_risk.predictors import PicklePredictor, validate_feature_frame


@pytest.fixture
def fitted_model_path(tmp_path):
    X = pd.DataFrame(
        {
            "feature_a": [0.0, 1.0, 0.0, 1.0],
            "feature_b": [0.0, 0.0, 1.0, 1.0],
        }
    )
    y = pd.Series([0, 0, 1, 1])
    model = LogisticRegression().fit(X, y)
    path = tmp_path / "model.pkl"
    with path.open("wb") as file:
        pickle.dump(model, file)
    return path, X


def test_persisted_predictor_loads_and_predicts(fitted_model_path):
    path, X = fitted_model_path
    predictor = PicklePredictor(path)
    probabilities = predictor.predict_proba(X)
    assert probabilities.shape == (4,)
    assert ((probabilities >= 0) & (probabilities <= 1)).all()


def test_validate_feature_frame_accepts_exact_columns(fitted_model_path):
    _, X = fitted_model_path
    assert validate_feature_frame(X, ["feature_a", "feature_b"]) is X


def test_predictor_rejects_incompatible_columns(fitted_model_path):
    path, X = fitted_model_path
    predictor = PicklePredictor(path)
    with pytest.raises(ValueError, match="exactly and in order"):
        predictor.predict_proba(X[["feature_b", "feature_a"]])
