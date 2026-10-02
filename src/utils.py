"""
utils.py
========
Reusable helper functions shared across notebooks, training scripts, and the
Streamlit dashboard.

WHY THIS FILE EXISTS
---------------------
Notebook 03 already defines an `evaluate_model()` function inline. That is
fine for a single notebook, but the moment the same evaluation logic is
needed again — in Notebook 04 for error analysis, or in the dashboard for
live predictions — copy-pasting it creates a maintenance risk: if you fix a
bug in one copy, the others silently stay wrong.

Moving shared logic into `src/utils.py` means every part of the project
imports the SAME function:

    from src.utils import evaluate_model, add_temperature_difference

This is what "production-grade" means in practice: not more complex code,
but code that has exactly one place where each idea lives.

Every function below reproduces logic that already exists and has already
been validated inside the completed notebooks — nothing new is introduced.
"""

from __future__ import annotations

from pathlib import Path
from typing import Any

import joblib
import numpy as np
import pandas as pd
from sklearn.metrics import (
    accuracy_score,
    f1_score,
    precision_score,
    recall_score,
    roc_auc_score,
)


# ==========================================================
# Feature Engineering
# ==========================================================
def add_temperature_difference(df: pd.DataFrame) -> pd.DataFrame:
    """
    Add the 'Temperature Difference [K]' engineering feature.

    Engineering rationale
    ----------------------
    Temperature Difference = Process Temperature - Air Temperature.

    This approximates the thermal gradient the machine has to dissipate
    into the surrounding air. A larger gradient means the process is
    generating heat faster than the ambient air can absorb it — a classic
    early-warning signal for Heat Dissipation Failure (HDF) in rotating
    industrial equipment, and directly analogous to thermal stress
    calculations used in chemical process equipment design.

    Parameters
    ----------
    df : pd.DataFrame
        DataFrame containing 'Process temperature [K]' and
        'Air temperature [K]' columns.

    Returns
    -------
    pd.DataFrame
        A COPY of the input DataFrame with the new column added.
        A copy is returned (rather than mutating in place) so callers never
        get surprised by a function silently changing their original data.
    """
    df = df.copy()
    df["Temperature Difference [K]"] = (
        df["Process temperature [K]"] - df["Air temperature [K]"]
    )
    return df


# ==========================================================
# Model Evaluation
# ==========================================================
def evaluate_model(
    model: Any, X_test: pd.DataFrame, y_test: pd.Series
) -> tuple[dict[str, float], np.ndarray, np.ndarray]:
    """
    Evaluate a trained binary classification model on a held-out test set.

    This is the exact evaluation function defined inline in
    notebooks/03_Model_Training.ipynb, moved here so Notebook 04, the
    dashboard, and any future script can all call the identical logic
    instead of re-implementing it.

    Parameters
    ----------
    model : fitted estimator (e.g. an sklearn Pipeline)
        Must implement .predict() and .predict_proba().
    X_test : pd.DataFrame
        Held-out feature matrix.
    y_test : pd.Series
        Held-out true labels.

    Returns
    -------
    metrics : dict
        Accuracy, Precision, Recall, F1 Score, ROC-AUC.
    y_pred : np.ndarray
        Hard class predictions (0 / 1).
    y_prob : np.ndarray
        Predicted probability of the positive class (failure = 1).
    """
    y_pred = model.predict(X_test)
    y_prob = model.predict_proba(X_test)[:, 1]

    metrics = {
        "Accuracy": accuracy_score(y_test, y_pred),
        "Precision": precision_score(y_test, y_pred),
        "Recall": recall_score(y_test, y_pred),
        "F1 Score": f1_score(y_test, y_pred),
        "ROC-AUC": roc_auc_score(y_test, y_prob),
    }

    return metrics, y_pred, y_prob


# ==========================================================
# Persistence
# ==========================================================
def save_pipeline(pipeline: Any, path: str | Path) -> None:
    """
    Save a fitted sklearn Pipeline (preprocessing + model) to disk with joblib.

    joblib is preferred over pickle for scikit-learn objects because it is
    more efficient at serializing large NumPy arrays, which Pipelines and
    tree-based models (Random Forest, XGBoost) contain in abundance.

    Parameters
    ----------
    pipeline : fitted sklearn Pipeline
    path : str or Path
        Destination file path (parent directories are created if missing).
    """
    path = Path(path)
    path.parent.mkdir(parents=True, exist_ok=True)
    joblib.dump(pipeline, path)


def load_pipeline(path: str | Path) -> Any:
    """
    Load a previously saved sklearn Pipeline from disk with joblib.

    Parameters
    ----------
    path : str or Path
        Path to a .pkl file created by `save_pipeline`.

    Returns
    -------
    The deserialized fitted pipeline, ready to call .predict() /
    .predict_proba() on new data.
    """
    return joblib.load(Path(path))


# ==========================================================
# Risk Categorization (used by the Streamlit dashboard, Stage 4)
# ==========================================================
def categorize_risk(probability: float, thresholds: dict[str, float]) -> str:
    """
    Convert a failure probability into a human-readable risk category.

    Parameters
    ----------
    probability : float
        Model's predicted probability of failure, in [0, 1].
    thresholds : dict
        Dict with 'low' and 'medium' keys, e.g. config.RISK_THRESHOLDS.

    Returns
    -------
    str
        One of "Low", "Medium", "High".
    """
    if probability < thresholds["low"]:
        return "Low"
    if probability < thresholds["medium"]:
        return "Medium"
    return "High"
