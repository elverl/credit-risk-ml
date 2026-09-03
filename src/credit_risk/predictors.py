"""Prediction components for credit-risk models."""

import pickle
from abc import ABC, abstractmethod

import numpy as np
import pandas as pd


def validate_feature_frame(
    X: pd.DataFrame,
    expected_features,
) -> pd.DataFrame:
    """Validate feature names and order for model inference."""
    if not isinstance(X, pd.DataFrame):
        raise TypeError("X must be a pandas DataFrame.")

    expected = list(expected_features)
    actual = X.columns.tolist()
    missing = [feature for feature in expected if feature not in actual]
    extra = [column for column in actual if column not in expected]

    if missing or extra or actual != expected:
        raise ValueError(
            "Input columns must match expected features exactly and in order. "
            f"Missing={missing}; Extra={extra}."
        )

    return X


class BasePredictor(ABC):
    """Base interface for credit-risk predictors."""

    @abstractmethod
    def predict_proba(self, X: pd.DataFrame) -> np.ndarray:
        """Return probability of default."""
        raise NotImplementedError

    @abstractmethod
    def load(self, path: str) -> None:
        """Load a trained model from disk."""
        raise NotImplementedError


class PicklePredictor(BasePredictor):
    """
    Credit-risk predictor based on a serialized binary classifier.
    """

    def __init__(self, model_path: str):
        """
        Parameters
        ----------
        model_path : str
            Path to a serialized model implementing predict_proba().
        """
        self.model_path = model_path
        self.model = None

        self.load(model_path)

    def load(self, path: str) -> None:
        """Load the model from a pickle file."""
        with open(path, "rb") as file:
            self.model = pickle.load(file)

    def predict_proba(self, X: pd.DataFrame) -> np.ndarray:
        """
        Return probability of default.

        Parameters
        ----------
        X : pandas.DataFrame
            Model features.

        Returns
        -------
        numpy.ndarray
            Probability of class 1 (default).
        """
        if self.model is None:
            raise ValueError(
                "Model has not been loaded."
            )

        expected_features = getattr(
            self.model,
            "feature_names_in_",
            None,
        )
        if expected_features is not None:
            validate_feature_frame(X, expected_features)

        classes = np.asarray(getattr(self.model, "classes_", []))
        positive_positions = np.flatnonzero(classes == 1)
        if len(positive_positions) != 1:
            raise ValueError(
                "The loaded model must contain exactly one "
                "positive class 1."
            )

        return self.model.predict_proba(X)[
            :,
            int(positive_positions[0]),
        ]


class LightGBMPredictor(PicklePredictor):
    """Backward-compatible alias for serialized LightGBM predictors."""
