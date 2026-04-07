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
import xgboost as xgb
from catboost import CatBoostClassifier
from sklearn.preprocessing import LabelEncoder, RobustScaler
from sklearn.ensemble import ExtraTreesClassifier

from prepare import load_data, get_cv_splits, evaluate

# ---------------------------------------------------------------------------
# Feature engineering (modify freely)
# ---------------------------------------------------------------------------

def create_features(df, feature_cols):
    """Create features from raw data. Returns (X, feature_names)."""
    X = df[feature_cols].copy()

    # Parse registration_date and extract date features
    X["registration_date"] = pd.to_datetime(X["registration_date"], errors="coerce")
    X["reg_year"] = X["registration_date"].dt.year
    X["reg_month"] = X["registration_date"].dt.month
    X["reg_dayofweek"] = X["registration_date"].dt.dayofweek
    X = X.drop(columns=["registration_date"])

    # Age feature
    X["age"] = 2026 - X["birth_year"]

    # Total spend and spend ratios
    spend_cols = ["spend_wines", "spend_fruits", "spend_meat", "spend_fish", "spend_sweets", "spend_gold"]
    X["total_spend"] = X[spend_cols].sum(axis=1)
    X["spend_wines_ratio"] = X["spend_wines"] / (X["total_spend"] + 1)
    X["spend_meat_ratio"] = X["spend_meat"] / (X["total_spend"] + 1)
    X["spend_gold_ratio"] = X["spend_gold"] / (X["total_spend"] + 1)

    # Total purchases
    purchase_cols = ["deals_purchases", "web_purchases", "catalog_purchases", "store_purchases"]
    X["total_purchases"] = X[purchase_cols].sum(axis=1)
    X["deals_ratio"] = X["deals_purchases"] / (X["total_purchases"] + 1)
    X["web_ratio"] = X["web_purchases"] / (X["total_purchases"] + 1)
    X["catalog_ratio"] = X["catalog_purchases"] / (X["total_purchases"] + 1)

    # Spend per purchase
    X["spend_per_purchase"] = X["total_spend"] / (X["total_purchases"] + 1)

    # Income per person in household
    X["num_family"] = X["num_children"] + X["num_teenagers"] + 1
    X["income_per_person"] = X["annual_income"] / (X["num_family"] + 1)

    # Interaction features
    X["age_x_income"] = X["age"] * X["annual_income"]
    X["spend_x_visits"] = X["total_spend"] * X["monthly_web_visits"]
    X["recency_x_spend"] = X["days_since_last_purchase"] * X["total_spend"]

    # Label encode all remaining object/string columns
    for col in X.columns:
        if X[col].dtype == object or str(X[col].dtype) == "str":
            le = LabelEncoder()
            X[col] = le.fit_transform(X[col].astype(str))

    # Drop id-like columns
    for col in ["customer_id"]:
        if col in X.columns:
            X = X.drop(columns=[col])

    feature_names = list(X.columns)
    return X, feature_names

# ---------------------------------------------------------------------------
# Model definition (modify freely)
# ---------------------------------------------------------------------------

LGB_PARAMS = {
    "objective": "binary",
    "metric": "auc",
    "verbosity": -1,
    "n_estimators": 2000,
    "learning_rate": 0.02,
    "num_leaves": 63,
    "max_depth": 7,
    "min_child_samples": 15,
    "subsample": 0.8,
    "subsample_freq": 1,
    "colsample_bytree": 0.8,
    "reg_alpha": 0.1,
    "reg_lambda": 1.0,
    "random_state": 42,
}

XGB_PARAMS = {
    "objective": "binary:logistic",
    "eval_metric": "auc",
    "verbosity": 0,
    "n_estimators": 2000,
    "learning_rate": 0.02,
    "max_depth": 6,
    "min_child_weight": 5,
    "gamma": 0.1,
    "subsample": 0.75,
    "colsample_bytree": 0.75,
    "colsample_bylevel": 0.75,
    "reg_alpha": 0.05,
    "reg_lambda": 0.5,
    "random_state": 42,
    "tree_method": "hist",
}

CB_PARAMS = {
    "iterations": 2000,
    "learning_rate": 0.02,
    "depth": 7,
    "l2_leaf_reg": 5,
    "min_data_in_leaf": 10,
    "loss_function": "Logloss",
    "eval_metric": "AUC",
    "random_seed": 123,
    "verbose": 0,
    "early_stopping_rounds": 50,
}

ET_PARAMS = {
    "n_estimators": 1000,
    "max_depth": 20,
    "min_samples_leaf": 3,
    "max_features": "sqrt",
    "random_state": 42,
    "n_jobs": -1,
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
    oof_preds_lgb = np.zeros(len(y))
    oof_preds_xgb = np.zeros(len(y))
    oof_preds_cb = np.zeros(len(y))
    oof_preds_et = np.zeros(len(y))
    fold_scores = []

    for fold_idx, (train_idx, val_idx) in enumerate(splits):
        X_train, X_val = X.iloc[train_idx], X.iloc[val_idx]
        y_train, y_val = y[train_idx], y[val_idx]

        # LightGBM
        lgb_model = lgb.LGBMClassifier(**LGB_PARAMS)
        lgb_model.fit(
            X_train, y_train,
            eval_set=[(X_val, y_val)],
            callbacks=[lgb.early_stopping(50, verbose=False)],
        )
        lgb_preds = lgb_model.predict_proba(X_val)[:, 1]
        oof_preds_lgb[val_idx] = lgb_preds

        # XGBoost
        xgb_params = {**XGB_PARAMS, "early_stopping_rounds": 50}
        xgb_model = xgb.XGBClassifier(**xgb_params)
        xgb_model.fit(
            X_train, y_train,
            eval_set=[(X_val, y_val)],
            verbose=False,
        )
        xgb_preds = xgb_model.predict_proba(X_val)[:, 1]
        oof_preds_xgb[val_idx] = xgb_preds

        # CatBoost
        cb_model = CatBoostClassifier(**CB_PARAMS)
        cb_model.fit(
            X_train, y_train,
            eval_set=(X_val, y_val),
        )
        cb_preds = cb_model.predict_proba(X_val)[:, 1]
        oof_preds_cb[val_idx] = cb_preds

        # ExtraTrees
        scaler = RobustScaler()
        X_train_sc = scaler.fit_transform(X_train)
        X_val_sc = scaler.transform(X_val)
        et_model = ExtraTreesClassifier(**ET_PARAMS)
        et_model.fit(X_train_sc, y_train)
        et_preds = et_model.predict_proba(X_val_sc)[:, 1]
        oof_preds_et[val_idx] = et_preds

        # Blend (LGB+XGB+CB+ET with 0.3/0.25/0.25/0.2)
        val_preds = 0.3 * lgb_preds + 0.25 * xgb_preds + 0.25 * cb_preds + 0.2 * et_preds
        fold_auc = evaluate(y_val, val_preds)
        fold_scores.append(fold_auc)
        print(f"Fold {fold_idx}: AUC = {fold_auc:.6f}")

    oof_preds = 0.3 * oof_preds_lgb + 0.25 * oof_preds_xgb + 0.25 * oof_preds_cb + 0.2 * oof_preds_et

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
