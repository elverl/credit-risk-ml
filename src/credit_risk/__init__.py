"""
Credit Risk ML package.

Reusable components for credit-risk prediction, evaluation,
threshold selection and explainability.
"""

from .evaluation import (
    evaluate_binary_classifier,
    ks_statistic,
    rank_models_by_global_score,
)
from .thresholds import threshold_youden
from .predictors import (
    BasePredictor,
    LightGBMPredictor,
    PicklePredictor,
    validate_feature_frame,
)
from .explainability import (
    BaseExplainer,
    CreditRiskExplainer,
    ExplanationContext,
    ExplanationResult,
    ShapContribution,
    compute_shap_explanation,
)
from .pipeline import CreditRiskPipeline

__all__ = [
    "evaluate_binary_classifier",
    "ks_statistic",
    "rank_models_by_global_score",
    "threshold_youden",
    "BasePredictor",
    "PicklePredictor",
    "validate_feature_frame",
    "LightGBMPredictor",
    "BaseExplainer",
    "CreditRiskExplainer",
    "ExplanationContext",
    "ExplanationResult",
    "ShapContribution",
    "compute_shap_explanation",
    "CreditRiskPipeline",
]
