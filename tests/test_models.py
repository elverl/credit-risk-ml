"""High-value checks for the fixed V2 model builders."""

from lightgbm import LGBMClassifier
from sklearn.ensemble import RandomForestClassifier
from sklearn.linear_model import LogisticRegression
from sklearn.pipeline import Pipeline
from xgboost import XGBClassifier

from credit_risk.models import (
    build_lightgbm,
    build_logistic_regression,
    build_random_forest,
    build_xgboost,
)


def test_logistic_regression_builder():
    model = build_logistic_regression()
    assert isinstance(model, Pipeline)
    estimator = model.named_steps["model"]
    assert isinstance(estimator, LogisticRegression)
    expected = {
        "C": 100.0,
        "class_weight": "balanced",
        "max_iter": 2000,
        "random_state": 123,
        "solver": "liblinear",
    }
    params = estimator.get_params()
    assert {key: params[key] for key in expected} == expected


def test_random_forest_builder():
    model = build_random_forest()
    assert isinstance(model, RandomForestClassifier)
    expected = {
        "n_estimators": 359,
        "criterion": "entropy",
        "max_depth": 21,
        "max_leaf_nodes": 60,
        "random_state": 123,
    }
    params = model.get_params()
    assert {key: params[key] for key in expected} == expected


def test_xgboost_builder():
    model = build_xgboost()
    assert isinstance(model, XGBClassifier)
    expected = {
        "n_estimators": 800,
        "grow_policy": "lossguide",
        "max_depth": 10,
        "max_leaves": 93,
        "random_state": 123,
    }
    params = model.get_params()
    assert {key: params[key] for key in expected} == expected


def test_lightgbm_builder():
    model = build_lightgbm()
    assert isinstance(model, LGBMClassifier)
    expected = {
        "n_estimators": 350,
        "max_depth": 13,
        "num_leaves": 102,
        "random_state": 123,
    }
    params = model.get_params()
    assert {key: params[key] for key in expected} == expected
