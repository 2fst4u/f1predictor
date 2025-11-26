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
    Train a model producing a "pace index" (lower is better/faster) per driver for the session.

    CRITICAL FIX: The form_index is already calculated as HIGHER = BETTER (more points, better position).
    We want to predict pace where LOWER = FASTER, so we negate it properly.

    Target: y = form_index (higher is better), but we'll invert the final predictions.

    Additionally, blend the learned prediction with a baseline from form/team_form/driver_team_form
    to prevent near-uniform outputs when per-event features are low-variance.
    """
    # CRITICAL FIX: form_index is HIGHER = BETTER (positive positions + points)
    # We want pace_index where LOWER = FASTER
    # So target should be: MINIMIZE (negative form)
    if "form_index" not in X.columns:
        # If somehow missing, create a neutral target (zeros), still return a trained baseline
        y = np.zeros(len(X), dtype=float)
    else:
        # FIXED: form_index is calculated as: -position + points (so HIGHER is better)
        # We want to predict something where LOWER is better (faster)
        # So we NEGATE it: y = -form_index
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

    # Choose model - FIXED: Better hyperparameters
    if _HAS_LGB:
        model = lgb.LGBMRegressor(
            n_estimators=200,  # Reduced for faster training
            learning_rate=0.1,  # Increased for stronger learning
            max_depth=5,  # Limited depth to prevent overfitting
            subsample=0.8,
            colsample_bytree=0.8,
            reg_alpha=0.1,  # L1 regularization
            reg_lambda=0.1,  # L2 regularization
            random_state=42,
            n_jobs=-1,
            verbose=-1
        )
    elif _HAS_XGB:
        model = xgb.XGBRegressor(
            n_estimators=200,
            max_depth=5,
            learning_rate=0.1,
            subsample=0.8,
            colsample_bytree=0.8,
            reg_lambda=1.0,
            random_state=42,
            tree_method="hist",
            n_jobs=-1,
            verbosity=0
        )
    else:
        model = GradientBoostingRegressor(
            n_estimators=100,
            learning_rate=0.1,
            max_depth=4,
            random_state=42
        )

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

    # FIXED: Baseline calculation
    # form_index is HIGHER = BETTER, so for pace (LOWER = FASTER), we negate it
    base = np.zeros(len(X), dtype=float)
    if "form_index" in X.columns:
        # Negate because form_index higher = better, but we want pace lower = faster
        base = base - X["form_index"].astype(float).values
    if "team_form_index" in X.columns:
        base = base - 0.5 * X["team_form_index"].astype(float).values
    if "driver_team_form_index" in X.columns:
        base = base - 0.3 * X["driver_team_form_index"].astype(float).values

    # If baseline has no variance, add tiny deterministic jitter for tie-breaks
    try:
        if np.nanstd(base) < 1e-9 and "driverId" in X.columns:
            h = pd.util.hash_pandas_object(X["driverId"], index=False).astype(np.uint64) % 997
            base = base + (h.astype(float) / 1e6)
    except Exception:
        pass

    # FIXED: Better blending strategy
    # Only blend if both have variance; otherwise use the one with variance
    yhat_std = float(np.nanstd(yhat))
    base_std = float(np.nanstd(base))

    if yhat_std < 1e-6 and base_std > 1e-6:
        # Model failed to learn, use baseline
        pace_hat = base
    elif base_std < 1e-6 and yhat_std > 1e-6:
        # Baseline is flat, use model
        pace_hat = yhat
    elif yhat_std > 1e-6 and base_std > 1e-6:
        # Both have signal, blend MORE towards model (it's trained on data)
        pace_hat = 0.75 * yhat + 0.25 * base  # FIXED: More weight to model
    else:
        # Both flat, use baseline with jitter
        pace_hat = base

    return pipe, pace_hat, features


def train_dnf_hazard_model(X: pd.DataFrame, hist: pd.DataFrame) -> Any:
    """
    Train a simple DNF probability model from historical race data.

    Features: driver/team DNF base rates + available weather features.
    Labels: proxy from base rates (since per-event labels aren't assembled here).
    Returns (clf, feature_columns, base_rates_df) or None if insufficient data.

    base_rates_df has columns: [driverId, constructorId, drv_dnf_rate, team_dnf_rate]
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

    # Join onto current X for training
    Xjoin = X.merge(base, on=["driverId", "constructorId"], how="left")
    Xjoin["drv_dnf_rate"] = Xjoin["drv_dnf_rate"].fillna(base["drv_dnf_rate"].mean())
    Xjoin["team_dnf_rate"] = Xjoin["team_dnf_rate"].fillna(base["team_dnf_rate"].mean())

    feat_cols = ["drv_dnf_rate", "team_dnf_rate"] + [c for c in Xjoin.columns if c.startswith("weather_")]

    if len(Xjoin) == 0:
        return None
    # Proxy label from relative risk (heuristic)
    threshold = float(Xjoin["drv_dnf_rate"].mean())
    y_proxy = (Xjoin["drv_dnf_rate"] * 0.5 + Xjoin["team_dnf_rate"] * 0.5 > threshold).astype(int)

    clf = GradientBoostingClassifier(random_state=42)
    clf.fit(Xjoin[feat_cols], y_proxy)

    # Return the base mapping so inference can recreate the same features
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
    clip_max: float = 0.35,
) -> np.ndarray:
    """
    Estimate per-driver DNF probabilities using Beta-smoothed empirical base rates.

    - Build DNF labels from historical race results.
    - Compute per-driver and per-team counts (k, n) and apply Beta smoothing:
        p = (k + alpha) / (n + alpha + beta)
    - Combine: p_dnf = driver_weight * p_driver + team_weight * p_team
    - Clip to [clip_min, clip_max] to avoid pathological extremes for tiny samples.

    Returns: numpy array aligned to current_X rows.
    """
    races = hist[hist["session"] == "race"].copy()
    if races.empty or current_X is None or current_X.empty:
        return np.full(len(current_X) if current_X is not None else 0, 0.1, dtype=float)

    status = races["status"].astype(str).str.lower()
    dnf = (~races["position"].notna()) | status.str.contains(
        "accident|engine|gear|suspension|electrical|hydraulics|dnf|brake|clutch"
    )
    races["dnf"] = dnf.astype(int)

    # Per-driver counts
    drv_counts = races.groupby("driverId")["dnf"].agg(["sum", "count"]).rename(columns={"sum": "k", "count": "n"})
    drv_counts["p"] = (drv_counts["k"] + alpha) / (drv_counts["n"] + alpha + beta)

    # Per-team counts
    team_counts = races.groupby("constructorId")["dnf"].agg(["sum", "count"]).rename(columns={"sum": "k", "count": "n"})
    team_counts["p"] = (team_counts["k"] + alpha) / (team_counts["n"] + alpha + beta)

    # Global fallback
    global_k = races["dnf"].sum()
    global_n = races.shape[0]
    global_p = (global_k + alpha) / (global_n + alpha + beta)

    # Map to current rows
    drv_map = drv_counts["p"].to_dict()
    team_map = team_counts["p"].to_dict()

    p_drv = current_X["driverId"].map(drv_map).astype(float)
    p_team = current_X["constructorId"].map(team_map).astype(float)

    p_drv = p_drv.fillna(global_p)
    p_team = p_team.fillna(global_p)

    p = driver_weight * p_drv.values + team_weight * p_team.values
    p = np.clip(p.astype(float), clip_min, clip_max)
    return p
