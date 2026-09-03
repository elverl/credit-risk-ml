"""
Data loading and dataset splitting utilities.
"""

from pathlib import Path

import pandas as pd
from sklearn.model_selection import train_test_split

from .config import FEATURES, RANDOM_STATE, TARGET


def load_modeling_data(path: str | Path) -> pd.DataFrame:
    """
    Load the modeling dataset from a CSV file.

    Parameters
    ----------
    path : str or pathlib.Path
        Path to the CSV dataset.

    Returns
    -------
    pandas.DataFrame
        Loaded dataset.
    """
    path = Path(path)

    if not path.exists():
        raise FileNotFoundError(f"Dataset not found: {path}")

    return pd.read_csv(path)


def select_modeling_variables(
    df: pd.DataFrame,
) -> tuple[pd.DataFrame, pd.Series]:
    """
    Select model features and target variable.

    Parameters
    ----------
    df : pandas.DataFrame
        Input modeling dataset.

    Returns
    -------
    tuple[pandas.DataFrame, pandas.Series]
        Feature matrix X and target vector y.
    """
    required_columns = FEATURES + [TARGET]

    missing_columns = [
        column
        for column in required_columns
        if column not in df.columns
    ]

    if missing_columns:
        raise ValueError(
            f"Missing required columns: {missing_columns}"
        )

    X = df[FEATURES].copy()
    y = df[TARGET].copy()

    return X, y


def create_train_test_split(
    df: pd.DataFrame,
    test_size: float = 0.20,
) -> tuple:
    """
    Create a stratified train/test split.

    Parameters
    ----------
    df : pandas.DataFrame
        Modeling dataset.

    test_size : float, default=0.20
        Proportion assigned to the test sample.

    Returns
    -------
    tuple
        X_train, X_test, y_train, y_test.
    """
    if not 0 < test_size < 1:
        raise ValueError("test_size must be between 0 and 1.")

    X, y = select_modeling_variables(df)

    return train_test_split(
        X,
        y,
        test_size=test_size,
        random_state=RANDOM_STATE,
        stratify=y,
    )