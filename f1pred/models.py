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
    Train a model producing a "pace index" (lower is better/faster) per driver.

    The form_index is calculated as: -position + points, so HIGHER form_index = BETTER driver.
    For pace (where LOWER = FASTER), we use: y = -form_index as the target.

    The model learns to predict pace from features, then we blend with a baseline
    derived from form indices to ensure robust predictions even with limited features.
    """
    # Target: form_index is HIGHER = BETTER, so negate for pace (LOWER = FASTER)
    if "form_index" not in X.columns:
        y = np.zeros(len(X), dtype=float)
    else:
        y = -X["form_index"].astype(float).values

    # Features (exclude identifiers, session meta, and target to prevent leakage)
    exclude_cols = [
        "driverId", "name", "code", "constructorId", "constructorName", "number",
        "session_type", "form_index"
    ]
    features, num_cols, cat_cols = _split_feature_columns(X, exclude=exclude_cols)

    # Build preprocessing pipeline
    num_pipe = Pipeline(steps=[
        ("imputer", SimpleImputer(strategy="median")),
        ("scaler", StandardScaler())
    ])

    cat_pipe = Pipeline(steps=[
        ("imputer", SimpleImputer(strategy="most_frequent")),
        ("onehot", OneHotEncoder(handle_unknown="ignore", sparse_output=False))
    ]) if cat_cols else "drop"

    pre = ColumnTransformer(transformers=[
        ("num", num_pipe, num_cols),
        ("cat", cat_pipe, cat_cols)
    ], remainder="drop")

    # Model selection with tuned hyperparameters
    if _HAS_LGB:
        model = lgb.LGBMRegressor(
            n_estimators=150,
            learning_rate=0.1,
            max_depth=4,
            num_leaves=15,
            subsample=0.8,
            colsample_bytree=0.8,
            reg_alpha=0.1,
            reg_lambda=0.1,
            random_state=42,
            n_jobs=-1,
            verbose=-1
        )
    elif _HAS_XGB:
        model = xgb.XGBRegressor(
            n_estimators=150,
            max_depth=4,
            learning_rate=0.1,
            subsample=0.8,
            colsample_bytree=0.8,
            reg_lambda=1.0,
            reg_alpha=0.1,
            random_state=42,
            tree_method="hist",
            n_jobs=-1,
            verbosity=0
        )
    else:
        model = GradientBoostingRegressor(
            n_estimators=100,
            learning_rate=0.1,
            max_depth=3,
            random_state=42
        )

    pipe = Pipeline(steps=[("pre", pre), ("model", model)])

    # Fit model
    Xmat = X[features].copy()
    if Xmat.shape[1] == 0:
        Xmat = pd.DataFrame({"const": np.ones(len(X), dtype=float)})
        pipe = Pipeline(steps=[("pre", "passthrough"), ("model", model)])
    
    pipe.fit(Xmat, y)
    yhat = pipe.predict(Xmat)

    # Build baseline from form indices (for robustness)
    # form_index is HIGHER = BETTER, so negate for pace (LOWER = FASTER)
    base = np.zeros(len(X), dtype=float)
    if "form_index" in X.columns:
        base = -X["form_index"].astype(float).values
    if "team_form_index" in X.columns:
        # team_form_index is points-based, HIGHER = BETTER, so negate
        base = base - 0.3 * X["team_form_index"].astype(float).values
    if "driver_team_form_index" in X.columns:
        base = base - 0.2 * X["driver_team_form_index"].astype(float).values

    # Add small jitter if baseline is flat (for tie-breaking)
    if np.nanstd(base) < 1e-9 and "driverId" in X.columns:
        try:
            h = pd.util.hash_pandas_object(X["driverId"], index=False).astype(np.uint64) % 997
            base = base + (h.astype(float) / 1e6)
        except Exception:
            pass

    # Blend model predictions with baseline
    yhat_std = float(np.nanstd(yhat))
    base_std = float(np.nanstd(base))

    if yhat_std < 1e-6 and base_std > 1e-6:
        # Model failed, use baseline only
        pace_hat = base
    elif base_std < 1e-6 and yhat_std > 1e-6:
        # Baseline is flat, use model only
        pace_hat = yhat
    elif yhat_std > 1e-6 and base_std > 1e-6:
        # Both have signal - weight towards model (it's trained on actual data)
        pace_hat = 0.7 * yhat + 0.3 * base
    else:
        # Both flat - use baseline with some noise
        pace_hat = base + np.random.RandomState(42).normal(0, 0.01, size=len(base))

    return pipe, pace_hat, features


def train_dnf_hazard_model(X: pd.DataFrame, hist: pd.DataFrame) -> Any:
    """
    Train a simple DNF probability model from historical race data.
    """
    races = hist[hist["session"] == "race"].copy()
    if races.empty:
        return None

    status = races["status"].astype(str).str.lower()
    dnf = (~races["position"].notna()) | status.str.contains(
        "accident|engine|gear|suspension|electrical|hydraulics|dnf|brake|clutch|collision|spin|damage"
    )
    races["dnf"] = dnf.astype(int)

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

    Xjoin = X.merge(base, on=["driverId", "constructorId"], how="left")
    Xjoin["drv_dnf_rate"] = Xjoin["drv_dnf_rate"].fillna(base["drv_dnf_rate"].mean())
    Xjoin["team_dnf_rate"] = Xjoin["team_dnf_rate"].fillna(base["team_dnf_rate"].mean())

    feat_cols = ["drv_dnf_rate", "team_dnf_rate"] + [c for c in Xjoin.columns if c.startswith("weather_")]

    if len(Xjoin) == 0:
        return None

    threshold = float(Xjoin["drv_dnf_rate"].mean())
    y_proxy = (Xjoin["drv_dnf_rate"] * 0.5 + Xjoin["team_dnf_rate"] * 0.5 > threshold).astype(int)

    clf = GradientBoostingClassifier(n_estimators=50, max_depth=3, random_state=42)
    clf.fit(Xjoin[feat_cols], y_proxy)

    base_out = base[["driverId", "constructorId", "drv_dnf_rate", "team_dnf_rate"]].copy()
    return (clf, feat_cols, base_out)


def estimate_dnf_probabilities(
    hist: pd.DataFrame,
    current_X: pd.DataFrame,
    alpha: float = 2.0,
    beta: float = 8.0,
    driver_weight: float = 0.6,
    team_weight: float = 0.4,
    clip_min: float = 0.02,
    clip_max: float = 0.30,
) -> np.ndarray:
    """
    Estimate per-driver DNF probabilities using Beta-smoothed empirical base rates.
    """
    races = hist[hist["session"] == "race"].copy()
    if races.empty or current_X is None or current_X.empty:
        return np.full(len(current_X) if current_X is not None else 0, 0.08, dtype=float)

    status = races["status"].astype(str).str.lower()
    dnf = (~races["position"].notna()) | status.str.contains(
        "accident|engine|gear|suspension|electrical|hydraulics|dnf|brake|clutch|collision|spin|damage"
    )
    races["dnf"] = dnf.astype(int)

    drv_counts = races.groupby("driverId")["dnf"].agg(["sum", "count"]).rename(columns={"sum": "k", "count": "n"})
    drv_counts["p"] = (drv_counts["k"] + alpha) / (drv_counts["n"] + alpha + beta)

    team_counts = races.groupby("constructorId")["dnf"].agg(["sum", "count"]).rename(columns={"sum": "k", "count": "n"})
    team_counts["p"] = (team_counts["k"] + alpha) / (team_counts["n"] + alpha + beta)

    global_k = races["dnf"].sum()
    global_n = races.shape[0]
    global_p = (global_k + alpha) / (global_n + alpha + beta)

    drv_map = drv_counts["p"].to_dict()
    team_map = team_counts["p"].to_dict()

    p_drv = current_X["driverId"].map(drv_map).astype(float).fillna(global_p)
    p_team = current_X["constructorId"].map(team_map).astype(float).fillna(global_p)

    p = driver_weight * p_drv.values + team_weight * p_team.values
    p = np.clip(p.astype(float), clip_min, clip_max)
    return p
