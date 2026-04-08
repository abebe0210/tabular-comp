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
from catboost import CatBoostClassifier

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

    # Age from birth_year (no clipping - let model handle outliers)
    X["age"] = 2024 - X["birth_year"]

    # Clip extreme income outliers (beyond 99th percentile)
    income_99 = X["annual_income"].quantile(0.99)
    X["annual_income"] = X["annual_income"].clip(upper=income_99)

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

    # Convert string columns to category type for CatBoost
    cat_cols = list(X.select_dtypes(include=["object"]).columns)
    for col in cat_cols:
        X[col] = X[col].astype(str).astype("category")

    # Frequency encoding for categorical columns (numeric version)
    for col in ["education_level", "marital_status"]:
        if col in X.columns:
            freq = X[col].value_counts(normalize=True)
            X[f"{col}_freq"] = X[col].map(freq).astype(float)

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

CAT_PARAMS = {
    "iterations": 1000,
    "learning_rate": 0.05,
    "depth": 6,
    "l2_leaf_reg": 3,
    "random_seed": 42,
    "verbose": 0,
    "eval_metric": "AUC",
    "early_stopping_rounds": 50,
}

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

BLEND_WEIGHT_CAT = 0.6
BLEND_WEIGHT_LGB = 0.4

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

    # Identify categorical feature indices for CatBoost
    cat_features = [i for i, col in enumerate(X.columns) if X[col].dtype.name == "category"]

    # Prepare LGB-compatible data (label encode categoricals)
    from sklearn.preprocessing import LabelEncoder
    X_lgb = X.copy()
    for col in X_lgb.select_dtypes(include=["category"]).columns:
        le = LabelEncoder()
        X_lgb[col] = le.fit_transform(X_lgb[col].astype(str))

    for fold_idx, (train_idx, val_idx) in enumerate(splits):
        X_train, X_val = X.iloc[train_idx], X.iloc[val_idx]
        X_train_lgb, X_val_lgb = X_lgb.iloc[train_idx], X_lgb.iloc[val_idx]
        y_train, y_val = y[train_idx], y[val_idx]

        # CatBoost
        cat_model = CatBoostClassifier(**CAT_PARAMS)
        cat_model.fit(X_train, y_train, eval_set=(X_val, y_val), cat_features=cat_features)
        cat_preds = cat_model.predict_proba(X_val)[:, 1]

        # LightGBM
        lgb_model = lgb.LGBMClassifier(**LGB_PARAMS)
        lgb_model.fit(X_train_lgb, y_train, eval_set=[(X_val_lgb, y_val)], callbacks=[lgb.early_stopping(50, verbose=False)])
        lgb_preds = lgb_model.predict_proba(X_val_lgb)[:, 1]

        # Blend
        val_preds = BLEND_WEIGHT_CAT * cat_preds + BLEND_WEIGHT_LGB * lgb_preds
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
