"""
Tabular competition training script.
This is the ONLY file the agent modifies.
Everything is fair game: features, model, hyperparameters, ensembles, etc.

Usage: uv run train.py
"""

import time
import numpy as np
import pandas as pd
import lightgbm as lgb

from prepare import load_data, get_cv_splits, evaluate

# ---------------------------------------------------------------------------
# Feature engineering (modify freely)
# ---------------------------------------------------------------------------

def create_features(df, feature_cols):
    """Create features from raw data. Returns (X, feature_names)."""
    X = df[feature_cols].copy()
    feature_names = list(X.columns)
    return X, feature_names

# ---------------------------------------------------------------------------
# Model definition (modify freely)
# ---------------------------------------------------------------------------

LGB_PARAMS = {
    "objective": "binary",
    "metric": "auc",
    "verbosity": -1,
    "n_estimators": 1000,
    "learning_rate": 0.05,
    "num_leaves": 31,
    "min_child_samples": 20,
    "subsample": 0.8,
    "colsample_bytree": 0.8,
    "random_state": 42,
}

# ---------------------------------------------------------------------------
# Training loop
# ---------------------------------------------------------------------------

def main():
    t_start = time.time()

    # Load data
    df, feature_cols, target_col = load_data()
    y = df[target_col].values

    # Feature engineering
    X, feature_names = create_features(df, feature_cols)

    # Cross-validation
    splits = get_cv_splits(y)
    oof_preds = np.zeros(len(y))
    fold_scores = []

    for fold_idx, (train_idx, val_idx) in enumerate(splits):
        X_train, X_val = X.iloc[train_idx], X.iloc[val_idx]
        y_train, y_val = y[train_idx], y[val_idx]

        model = lgb.LGBMClassifier(**LGB_PARAMS)
        model.fit(
            X_train, y_train,
            eval_set=[(X_val, y_val)],
            callbacks=[lgb.early_stopping(50, verbose=False)],
        )

        val_preds = model.predict_proba(X_val)[:, 1]
        oof_preds[val_idx] = val_preds
        fold_auc = evaluate(y_val, val_preds)
        fold_scores.append(fold_auc)
        print(f"Fold {fold_idx}: AUC = {fold_auc:.6f}")

    # Overall CV score
    overall_auc = evaluate(y, oof_preds)
    elapsed = time.time() - t_start

    # Output in fixed format (agent parses these lines)
    print("---")
    print(f"val_auc:          {overall_auc:.6f}")
    print(f"mean_fold_auc:    {np.mean(fold_scores):.6f}")
    print(f"std_fold_auc:     {np.std(fold_scores):.6f}")
    print(f"elapsed_seconds:  {elapsed:.1f}")
    print(f"n_features:       {len(feature_names)}")
    print(f"n_samples:        {len(y)}")

if __name__ == "__main__":
    main()
