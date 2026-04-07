"""
Fixed evaluation framework for tabular data competition.
DO NOT MODIFY this file — it contains the ground truth evaluation.

Usage:
    from prepare import load_data, get_cv_splits, evaluate, TARGET_COL
"""

import os
import numpy as np
import pandas as pd
from sklearn.model_selection import StratifiedKFold
from sklearn.metrics import roc_auc_score

# ---------------------------------------------------------------------------
# Configuration (edit these for your competition)
# ---------------------------------------------------------------------------

TARGET_COL = "target"           # target column name
DATA_DIR = os.path.join(os.path.dirname(__file__), "data")
N_SPLITS = 10                    # number of CV folds
RANDOM_STATE = 42

# ---------------------------------------------------------------------------
# Data loading
# ---------------------------------------------------------------------------

def load_data():
    """Load training data. Returns (df, feature_cols, target_col)."""
    train_path = os.path.join(DATA_DIR, "train.csv")
    if not os.path.exists(train_path):
        raise FileNotFoundError(
            f"Training data not found at {train_path}. "
            f"Place your train.csv in the data/ directory."
        )
    df = pd.read_csv(train_path)
    if TARGET_COL not in df.columns:
        raise ValueError(
            f"Target column '{TARGET_COL}' not found in data. "
            f"Available columns: {list(df.columns)}"
        )
    feature_cols = [c for c in df.columns if c != TARGET_COL]
    return df, feature_cols, TARGET_COL


def load_test_data():
    """Load test data for submission. Returns DataFrame or None."""
    test_path = os.path.join(DATA_DIR, "test.csv")
    if not os.path.exists(test_path):
        return None
    return pd.read_csv(test_path)

# ---------------------------------------------------------------------------
# Cross-validation
# ---------------------------------------------------------------------------

def get_cv_splits(y, n_splits=N_SPLITS, random_state=RANDOM_STATE):
    """Return StratifiedKFold splits as list of (train_idx, val_idx)."""
    skf = StratifiedKFold(n_splits=n_splits, shuffle=True, random_state=random_state)
    return list(skf.split(np.zeros(len(y)), y))

# ---------------------------------------------------------------------------
# Evaluation (DO NOT CHANGE — this is the fixed metric)
# ---------------------------------------------------------------------------

def evaluate(y_true, y_pred_proba):
    """
    Compute ROC-AUC score.
    y_pred_proba: predicted probabilities for the positive class.
    Returns float (higher is better).
    """
    return roc_auc_score(y_true, y_pred_proba)
