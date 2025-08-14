from __future__ import annotations
from typing import Tuple, Any, List
import numpy as np
import pandas as pd

from sklearn.preprocessing import StandardScaler, OneHotEncoder
from sklearn.pipeline import Pipeline
from sklearn.compose import ColumnTransformer
from sklearn.impute import SimpleImputer
from sklearn.ensemble import GradientBoostingRegressor, GradientBoostingClassifier


# Try optional boosters; fall back to sklearn GBM
try:
    import lightgbm as lgb
    _HAS_LGB = True
except Exception:
    _HAS_LGB = False

try:
    import xgboost as xgb
    _HAS_XGB = True
except Exception:
    _HAS_XGB = False


def _split_feature_columns(X: pd.DataFrame, exclude: List[str]) -> Tuple[List[str], List[str], List[str]]:
    """
    Returns (all_features, numeric_cols, categorical_cols), excluding provided columns.
    """
    features = [c for c in X.columns if c not in exclude]
    cat_cols = [c for c in features if X[c].dtype == "object"]
    num_cols = [c for c in features if c not in cat_cols]
    return features, num_cols, cat_cols


def train_pace_model(X: pd.DataFrame, session_type: str) -> Tuple[Any, np.ndarray, list]:
    """
    Train a model producing a "pace index" (lower is better) per driver for the session.

    Target: y = -form_index (so lower predicted value = faster).
    To avoid target leakage, 'form_index' is excluded from the feature set.
    """
    # Target
    if "form_index" not in X.columns:
        # If somehow missing, create a neutral target (zeros), still return a trained baseline
        y = np.zeros(len(X), dtype=float)
    else:
        y = -X["form_index"].astype(float).values

    # Features (exclude identifiers, session meta, and target)
    exclude_cols = [
        "driverId", "name", "code", "constructorId", "constructorName", "number",
        "session_type", "form_index"  # prevent leakage
    ]
    features, num_cols, cat_cols = _split_feature_columns(X, exclude=exclude_cols)

    # Build preprocessing
    num_pipe = Pipeline(steps=[
        ("imputer", SimpleImputer(strategy="median")),
        ("scaler", StandardScaler())
    ])

    # If there are any categorical columns, one-hot encode them robustly
    cat_pipe = Pipeline(steps=[
        ("imputer", SimpleImputer(strategy="most_frequent")),
        ("onehot", OneHotEncoder(handle_unknown="ignore", sparse_output=False))
    ]) if cat_cols else "drop"

    pre = ColumnTransformer(transformers=[
        ("num", num_pipe, num_cols),
        ("cat", cat_pipe, cat_cols)
    ], remainder="drop")

    # Choose model
    if _HAS_LGB:
        model = lgb.LGBMRegressor(
            n_estimators=400,
            learning_rate=0.05,
            max_depth=-1,
            subsample=0.8,
            colsample_bytree=0.8,
            random_state=42,
            n_jobs=-1,
            verbose=-1
        )
    elif _HAS_XGB:
        model = xgb.XGBRegressor(
            n_estimators=600,
            max_depth=6,
            learning_rate=0.05,
            subsample=0.8,
            colsample_bytree=0.8,
            reg_lambda=1.0,
            random_state=42,
            tree_method="hist",
            n_jobs=-1,
            verbosity=0
        )
    else:
        model = GradientBoostingRegressor(random_state=42)

    pipe = Pipeline(steps=[("pre", pre), ("model", model)])

    # Fit; guard against empty feature matrix
    Xmat = X[features].copy()
    if Xmat.shape[1] == 0:
        # No usable features; fallback to predicting the mean target
        # Train a minimal model on a single constant feature
        Xmat = pd.DataFrame({"const": np.ones(len(X), dtype=float)})
        pre_fallback = "passthrough"
        pipe = Pipeline(steps=[("pre", pre_fallback), ("model", model)])
    pipe.fit(Xmat, y)
    yhat = pipe.predict(Xmat)

    return pipe, yhat, features


def train_dnf_hazard_model(X: pd.DataFrame, hist: pd.DataFrame) -> Any:
    """
    Train a simple DNF probability model from historical race data.

    Features: driver/team DNF base rates + available weather features.
    Labels: proxy from base rates (since per-event labels aren't assembled here).
    Returns (clf, feature_columns) or None if insufficient data.
    """
    races = hist[hist["session"] == "race"].copy()
    if races.empty:
        return None

    # Label DNF: non-finish statuses
    status = races["status"].astype(str).str.lower()
    dnf = (~races["position"].notna()) | status.str.contains(
        "accident|engine|gear|suspension|electrical|hydraulics|dnf|brake|clutch"
    )
    races["dnf"] = dnf.astype(int)

    # Base rates
    dnf_rate_driver = races.groupby("driverId")["dnf"].mean().rename("drv_dnf_rate")
    dnf_rate_team = races.groupby("constructorId")["dnf"].mean().rename("team_dnf_rate")

    base = races[["driverId", "constructorId"]].drop_duplicates().merge(
        dnf_rate_driver, on="driverId", how="left"
    ).merge(
        dnf_rate_team, on="constructorId", how="left"
    )

    if base.empty:
        return None

    base["drv_dnf_rate"] = base["drv_dnf_rate"].fillna(base["drv_dnf_rate"].mean())
    base["team_dnf_rate"] = base["team_dnf_rate"].fillna(base["team_dnf_rate"].mean())

    # Join onto current X
    Xjoin = X.merge(base, on=["driverId", "constructorId"], how="left")
    Xjoin["drv_dnf_rate"] = Xjoin["drv_dnf_rate"].fillna(Xjoin["drv_dnf_rate"].mean())
    Xjoin["team_dnf_rate"] = Xjoin["team_dnf_rate"].fillna(Xjoin["team_dnf_rate"].mean())

    feat_cols = ["drv_dnf_rate", "team_dnf_rate"] + [c for c in Xjoin.columns if c.startswith("weather_")]

    # Build a proxy label from relative risk to let the classifier calibrate probabilities
    # This is a heuristic; a full implementation would build event-level labels.
    if len(Xjoin) == 0:
        return None
    threshold = float(Xjoin["drv_dnf_rate"].mean())
    y_proxy = (Xjoin["drv_dnf_rate"] * 0.5 + Xjoin["team_dnf_rate"] * 0.5 > threshold).astype(int)

    clf = GradientBoostingClassifier(random_state=42)
    clf.fit(Xjoin[feat_cols], y_proxy)
    return (clf, feat_cols)