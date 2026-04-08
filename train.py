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

SPEND_COLS = ["spend_wines", "spend_fruits", "spend_meat", "spend_fish", "spend_sweets", "spend_gold"]
PURCHASE_COLS = ["deals_purchases", "web_purchases", "catalog_purchases", "store_purchases"]


def create_features(df, feature_cols):
    """Create features from raw data. Returns (X, feature_names)."""
    X = df[feature_cols].copy()

    # Drop ID column
    if "customer_id" in X.columns:
        X = X.drop(columns=["customer_id"])

    # Age from birth_year
    X["age"] = 2024 - X["birth_year"]

    # Total spending and purchases
    X["total_spend"] = X[SPEND_COLS].sum(axis=1)
    X["total_purchases"] = X[PURCHASE_COLS].sum(axis=1)

    # Spending ratios
    total_spend = X["total_spend"].replace(0, 1)
    for col in SPEND_COLS:
        X[f"{col}_ratio"] = X[col] / total_spend

    # Purchases ratios
    total_purch = X["total_purchases"].replace(0, 1)
    for col in PURCHASE_COLS:
        X[f"{col}_ratio"] = X[col] / total_purch

    # Income per household member
    X["household_size"] = 1 + X["num_children"] + X["num_teenagers"]
    X["income_per_member"] = X["annual_income"] / X["household_size"]

    # Spend per purchase
    X["spend_per_purchase"] = X["total_spend"] / total_purch

    # Days since registration from registration_date
    X["reg_date"] = pd.to_datetime(X["registration_date"], errors="coerce")
    ref_date = pd.Timestamp("2024-01-01")
    X["days_since_reg"] = (ref_date - X["reg_date"]).dt.days
    X = X.drop(columns=["registration_date", "reg_date"])

    # Label encode remaining string columns
    from sklearn.preprocessing import LabelEncoder
    for col in X.select_dtypes(include=["object"]).columns:
        le = LabelEncoder()
        X[col] = le.fit_transform(X[col].astype(str))

    # Frequency encoding for categorical columns
    for col in ["education_level", "marital_status"]:
        if col in X.columns:
            freq = X[col].value_counts(normalize=True)
            X[f"{col}_freq"] = X[col].map(freq)

    # Missing income flag
    X["income_missing"] = X["annual_income"].isna().astype(int)

    # Log income (skewed)
    X["log_income"] = np.log1p(X["annual_income"].fillna(0))

    # Log total spend
    X["log_total_spend"] = np.log1p(X["total_spend"])

    # Has children flag
    X["has_children"] = (X["num_children"] + X["num_teenagers"] > 0).astype(int)

    # Fill NaN
    X = X.fillna(-999)

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
