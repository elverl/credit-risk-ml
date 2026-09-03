"""
Model definitions for the credit-risk project.
"""

from lightgbm import LGBMClassifier
from sklearn.ensemble import RandomForestClassifier
from sklearn.linear_model import LogisticRegression
from sklearn.pipeline import Pipeline
from sklearn.preprocessing import StandardScaler
from xgboost import XGBClassifier


def build_logistic_regression() -> Pipeline:
    """
    Build the logistic-regression baseline model.

    Returns
    -------
    sklearn.pipeline.Pipeline
        Pipeline containing standardization and logistic regression.
    """
    return Pipeline(
        steps=[
            (
                "scaler",
                StandardScaler(),
            ),
            (
                "model",
                LogisticRegression(
                    C=100.0,
                    class_weight="balanced",
                    max_iter=2000,
                    random_state=123,
                    solver="liblinear",
                ),
            ),
        ]
    )


def build_random_forest() -> RandomForestClassifier:
    """Build the fixed Random Forest specification for V2."""
    return RandomForestClassifier(
        ccp_alpha=1e-06,
        class_weight="balanced",
        criterion="entropy",
        max_depth=21,
        max_leaf_nodes=60,
        max_samples=0.5,
        min_impurity_decrease=2.7152581234867345e-05,
        min_samples_leaf=72,
        min_samples_split=20,
        n_estimators=359,
        n_jobs=1,
        oob_score=True,
        random_state=123,
    )


def build_xgboost() -> XGBClassifier:
    """Build the fixed XGBoost specification for V2."""
    return XGBClassifier(
        colsample_bylevel=1.0,
        colsample_bynode=0.9337577656962762,
        colsample_bytree=0.8673942909590461,
        gamma=1.2532744673554702e-05,
        grow_policy="lossguide",
        learning_rate=0.050155030302911026,
        max_depth=10,
        max_leaves=93,
        min_child_weight=1.8403200343739003,
        n_estimators=800,
        reg_alpha=49.99999999999999,
        reg_lambda=10.0,
        scale_pos_weight=1.014471923705631,
        subsample=0.8438022389486756,
        random_state=123,
        n_jobs=1,
    )


def build_lightgbm() -> LGBMClassifier:
    """
    Build the LightGBM credit-risk model.

    Returns
    -------
    lightgbm.LGBMClassifier
        Configured LightGBM classifier.
    """
    return LGBMClassifier(
        colsample_bytree=0.6305920562163216,
        importance_type="gain",
        learning_rate=0.04207581459727521,
        max_depth=13,
        min_child_samples=65,
        min_child_weight=0.011714037545664444,
        min_split_gain=0.04554283774603906,
        n_estimators=350,
        n_jobs=-1,
        num_leaves=102,
        random_state=123,
        reg_alpha=0.01893089141356978,
        reg_lambda=0.000582205223528807,
        subsample=0.7433075514417062,
        subsample_for_bin=121503,
        subsample_freq=5,
    )
