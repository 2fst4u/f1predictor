"""Machine learning models for F1 predictions.

This module provides pace prediction and DNF probability estimation using
gradient boosting models. Supports LightGBM, XGBoost, or sklearn fallback.
"""
from __future__ import annotations
from typing import Tuple, Any, List, Optional, Dict, TYPE_CHECKING
import warnings

if TYPE_CHECKING:
    import pandas as pd
    import numpy as np
    from datetime import datetime

from .util import get_logger

# Suppress harmless sklearn warning about feature names when using preprocessing pipelines
warnings.filterwarnings("ignore", message="X does not have valid feature names")

__all__ = [
    "train_pace_model",
    "estimate_dnf_probabilities",
    "compute_shap_values",
]

logger = get_logger(__name__)


def build_hist_training_X(hist: 'pd.DataFrame', X_current: 'pd.DataFrame',
                          ref_date: 'datetime', half_life_days: int = 120,
                          max_events: int = 200,
                          boost_factor: float = 1.0,
                          qual_boost_factor: float = 1.0) -> Optional['pd.DataFrame']:
    """Build a lightweight training DataFrame from historical race results.

    Uses only columns that overlap with the current event feature matrix ``X_current``
    so that the GBM pipeline can be trained out-of-sample and then predict on
    ``X_current`` without column mismatch.

    The per-driver ``form_index`` is computed at each historical event's date so the
    GBM can learn the relationship between features and outcome quality *across*
    many events rather than memorising a single grid.

    Returns ``None`` if insufficient history is available.
    """
    import numpy as np
    import pandas as pd

    if hist is None or hist.empty:
        return None

    # Include both race and sprint for general pace training
    # For features like DNF rate, we need the full history including non-finishes
    races_full = hist[hist["session"].isin(["race", "sprint"])].dropna(subset=["driverId", "date"]).copy()
    if len(races_full) < 40:  # not enough data to meaningfully train
        return None

    # For the target variable (pace), we only use finishes
    races = races_full.dropna(subset=["position"]).copy()
    if len(races) < 20:
        return None

    races["points"] = races["points"].fillna(0.0)
    races_full["points"] = races_full["points"].fillna(0.0)

    # Ensure circuitId exists (might be missing in mocked historical data)
    if "circuitId" not in races_full.columns:
        races_full = races_full.assign(circuitId=None)
    if "circuitId" not in races.columns:
        races = races.assign(circuitId=None)

    # Identify the most recent events (by unique (season, round) pairs)
    event_keys = races_full[["season", "round", "date", "circuitId"]].drop_duplicates().sort_values("date")
    event_keys = event_keys.tail(max_events)

    rows = []

    # ⚡ Bolt: Pre-extract numpy arrays to avoid pandas overhead in the loop
    # Factorize driver and team IDs once to use np.bincount for aggregations
    driver_ids_full = races_full["driverId"].values
    d_codes_full, uniques = pd.factorize(driver_ids_full)
    n_drivers = len(uniques)

    # Map races finishes back to these codes
    d_codes = pd.Series(range(n_drivers), index=uniques).reindex(races["driverId"]).values

    # Safely extract constructorId
    if "constructorId" in races_full.columns:
        races_cons_full = races_full["constructorId"].values
    else:
        races_cons_full = np.array([None] * len(races_full))

    team_ids_full = races_cons_full
    t_codes_full, t_uniques = pd.factorize(team_ids_full)
    n_teams = len(t_uniques)

    # Map races finishes back to team codes
    # Ensure constructorId exists in races
    if "constructorId" not in races.columns:
        races = races.assign(constructorId=None)
    t_codes = pd.Series(range(n_teams), index=t_uniques).reindex(races["constructorId"]).fillna(-1).astype(int).values

    # Pre-calculate base values (position and points) for races (finishes)
    races_pos = races["position"].values.astype(float)
    races_pts = races["points"].values.astype(float)
    races_val_base = -races_pos + races_pts

    # Pre-calculate DNF status from races_full
    if "is_dnf" in races_full.columns:
        r_full_dnf = races_full["is_dnf"].astype(float).values
    else:
        rf_pos = races_full["position"].values
        rf_status = races_full["status"].astype(str).str.lower().values if "status" in races_full.columns else np.array(["finished"] * len(races_full))
        # Use a vectorized check for DNF status matching features.py
        is_dnf_status = pd.Series(rf_status).str.contains(
            "accident|engine|gear|suspension|electrical|hydraulics|dnf|brake|clutch|collision|spin|damage"
        ).values
        r_full_dnf = ((pd.isna(rf_pos.astype(float))) | is_dnf_status).astype(float)


    # Extract keys and other required data
    races_dates = pd.to_datetime(races["date"], utc=True).values
    races_full_dates = pd.to_datetime(races_full["date"], utc=True).values
    races_full_circuits = races_full["circuitId"].values
    races_seasons = races["season"].values
    races_rounds = races["round"].values
    races_sessions = races["session"].values
    races_circuits = races["circuitId"].values

    # Extract qualifying data for qualifying_form_index calculation
    # Ensure qpos column exists for dropna subset check
    if "qpos" not in hist.columns:
        hist = hist.assign(qpos=np.nan)

    qual = hist[hist["session"].isin(["qualifying", "sprint_qualifying"])].dropna(subset=["driverId", "qpos", "date"]).copy()
    if not qual.empty and "constructorId" in qual.columns:
        qual["team_avg"] = qual.groupby(["season", "round", "constructorId"])["qpos"].transform("mean")
        qual["q_delta"] = qual["team_avg"] - qual["qpos"]
    else:
        qual["q_delta"] = 0.0

    qual_driver_ids = qual["driverId"].values
    q_dcodes = pd.Series(range(n_drivers), index=uniques).reindex(qual_driver_ids).fillna(-1).astype(int).values
    qual_pos = qual["qpos"].values.astype(float)
    qual_delta = qual["q_delta"].values.astype(float)
    qual_dates = pd.to_datetime(qual["date"], utc=True).values
    qual_seasons = qual["season"].values
    qual_sessions = qual["session"].values

    # Handle NaN grids gracefully in numpy arrays
    races_grid = races["grid"].values
    races_grid_float = np.array([float(x) if pd.notna(x) else np.nan for x in races_grid])

    # O(N) loop over events, using highly vectorized numpy operations
    for evt_row in event_keys.itertuples(index=False):
        s = getattr(evt_row, "season")
        r = getattr(evt_row, "round")
        d = getattr(evt_row, "date")

        d_val = pd.Timestamp(d).to_datetime64()

        # Mask for current event
        evt_mask = (races_seasons == s) & (races_rounds == r)
        if not np.any(evt_mask):
            continue

        # Mask for prior history (strictly before the event date)
        prior_mask = races_dates < d_val
        if not np.any(prior_mask):
            continue

        # Extract prior history arrays
        p_dates = races_dates[prior_mask]
        p_season = races_seasons[prior_mask]
        p_val = races_val_base[prior_mask]
        p_dcodes = d_codes[prior_mask]

        # Calculate exponential recency weights
        diff = d_val - p_dates
        ages = diff.astype('timedelta64[ns]').astype(float) / 86400000000000.0
        w = np.exp2(-ages / max(1.0, float(half_life_days)))

        # Aggregate by driver using pure numpy bincount (~10x faster than pandas groupby)
        # Apply session-specific boosts within the current season
        p_sess = races_sessions[prior_mask]

        # Boost current season races
        is_p_cur_race = (p_season == s) & (p_sess == "race")
        if np.any(is_p_cur_race):
            w[is_p_cur_race] *= boost_factor

        # Boost current season sprints (use qual boost for sprints to match features.py)
        is_p_cur_sprint = (p_season == s) & (p_sess == "sprint")
        if np.any(is_p_cur_sprint):
            w[is_p_cur_sprint] *= qual_boost_factor

        wval = p_val * w
        w_sum = np.bincount(p_dcodes, weights=w, minlength=n_drivers)
        wval_sum = np.bincount(p_dcodes, weights=wval, minlength=n_drivers)


        # Compute form index safely
        valid = w_sum > 0
        form_index = np.zeros(n_drivers, dtype=float)
        form_index[valid] = wval_sum[valid] / np.maximum(w_sum[valid], 1e-6)

        # --- Calculate sprint_form_index ---
        is_sprint = (p_sess == "sprint")
        sprint_form_index = np.zeros(n_drivers, dtype=float)
        valid_sprint = np.zeros(n_drivers, dtype=bool)
        if np.any(is_sprint):
            sw_sum = np.bincount(p_dcodes[is_sprint], weights=w[is_sprint], minlength=n_drivers)
            swval_sum = np.bincount(p_dcodes[is_sprint], weights=wval[is_sprint], minlength=n_drivers)
            valid_sprint = sw_sum > 0
            sprint_form_index[valid_sprint] = swval_sum[valid_sprint] / np.maximum(sw_sum[valid_sprint], 1e-6)


        # --- Calculate team_form_index ---
        p_pts = races_pts[prior_mask]
        p_t_codes = t_codes[prior_mask]
        t_valid_mask = p_t_codes >= 0
        tw_sum = np.bincount(p_t_codes[t_valid_mask], weights=w[t_valid_mask], minlength=n_teams)
        twval_sum = np.bincount(p_t_codes[t_valid_mask], weights=(p_pts[t_valid_mask] * w[t_valid_mask]), minlength=n_teams)
        t_valid = tw_sum > 0
        team_form_index = np.zeros(n_teams, dtype=float)
        team_form_index[t_valid] = twval_sum[t_valid] / np.maximum(tw_sum[t_valid], 1e-6)

        # --- Calculate grid_finish_delta (racecraft) ---
        # gain = grid - position
        p_gains = races_grid_float[prior_mask] - races_pos[prior_mask]
        valid_gain = ~np.isnan(p_gains)
        gf_index = np.zeros(n_drivers, dtype=float)
        if np.any(valid_gain):
            gw_sum = np.bincount(p_dcodes[valid_gain], weights=w[valid_gain], minlength=n_drivers)
            gwval_sum = np.bincount(p_dcodes[valid_gain], weights=(p_gains[valid_gain] * w[valid_gain]), minlength=n_drivers)
            g_valid = gw_sum > 0
            gf_index[g_valid] = gwval_sum[g_valid] / np.maximum(gw_sum[g_valid], 1e-6)


        # --- Calculate circuit proficiency (using races_full for DNF rate) ---
        cid = getattr(evt_row, "circuitId")
        p_full_mask = races_full_dates < d_val
        c_mask = p_full_mask & (races_full_circuits == cid)

        c_avg_pos = np.full(n_drivers, 15.0, dtype=float)
        c_dnf_rate = np.zeros(n_drivers, dtype=float)
        c_exp = np.zeros(n_drivers, dtype=float)

        if np.any(c_mask):
            p_full_dcodes_masked = d_codes_full[c_mask]
            p_full_dnf_masked = r_full_dnf[c_mask]

            cw_sum = np.bincount(p_full_dcodes_masked, minlength=n_drivers)
            cd_sum = np.bincount(p_full_dcodes_masked, weights=p_full_dnf_masked, minlength=n_drivers)
            c_valid = cw_sum > 0
            c_dnf_rate[c_valid] = cd_sum[c_valid] / np.maximum(cw_sum[c_valid], 1e-6)
            c_exp[c_valid] = cw_sum[c_valid]

        # Average position needs finishes only
        c_fin_mask = (races_dates < d_val) & (races_circuits == cid)
        if np.any(c_fin_mask):
            p_fin_dcodes_masked = d_codes[c_fin_mask]
            cwf_sum = np.bincount(p_fin_dcodes_masked, minlength=n_drivers)
            cwfval_sum = np.bincount(p_fin_dcodes_masked, weights=races_pos[c_fin_mask], minlength=n_drivers)
            cf_valid = cwf_sum > 0
            c_avg_pos[cf_valid] = cwfval_sum[cf_valid] / np.maximum(cwf_sum[cf_valid], 1e-6)


        # --- Calculate qualifying_form_index & teammate_delta ---
        q_form_index = np.zeros(n_drivers, dtype=float)
        tm_delta_index = np.zeros(n_drivers, dtype=float)
        sprint_q_form_index = np.zeros(n_drivers, dtype=float)
        valid_sq = np.zeros(n_drivers, dtype=bool)

        prior_q_mask = qual_dates < d_val
        if np.any(prior_q_mask):
            pq_dates = qual_dates[prior_q_mask]
            pq_season = qual_seasons[prior_q_mask]
            pq_pos = qual_pos[prior_q_mask]
            pq_qdelta = qual_delta[prior_q_mask]
            pq_dcodes_sub = q_dcodes[prior_q_mask]

            # Remove any drivers not in our main unique set
            valid_pq = pq_dcodes_sub >= 0
            pq_dates = pq_dates[valid_pq]
            pq_season = pq_season[valid_pq]
            pq_pos = pq_pos[valid_pq]
            pq_qdelta = pq_qdelta[valid_pq]
            pq_dcodes_sub = pq_dcodes_sub[valid_pq]

            if len(pq_pos) > 0:
                diff_q = d_val - pq_dates
                ages_q = diff_q.astype('timedelta64[ns]').astype(float) / 86400000000000.0
                wq = np.exp2(-ages_q / max(1.0, float(half_life_days)))

                # Apply qual boost for current season
                boost_q_mask = pq_season == s
                wq[boost_q_mask] *= qual_boost_factor

                wqval = -pq_pos * wq
                wqval_delta = pq_qdelta * wq
                wq_sum = np.bincount(pq_dcodes_sub, weights=wq, minlength=n_drivers)
                wqval_sum = np.bincount(pq_dcodes_sub, weights=wqval, minlength=n_drivers)
                wqdelta_sum = np.bincount(pq_dcodes_sub, weights=wqval_delta, minlength=n_drivers)


                valid_q = wq_sum > 0
                q_form_index[valid_q] = wval_sum_val = wqval_sum[valid_q] / np.maximum(wq_sum[valid_q], 1e-6)
                tm_delta_index[valid_q] = wqdelta_sum[valid_q] / np.maximum(wq_sum[valid_q], 1e-6)

                # --- Calculate sprint_qualifying_form_index ---
                pq_sessions = qual_sessions[prior_q_mask][valid_pq]
                is_sq = (pq_sessions == "sprint_qualifying")
                sprint_q_form_index = np.zeros(n_drivers, dtype=float)
                valid_sq = np.zeros(n_drivers, dtype=bool)
                if np.any(is_sq):
                    sqw_sum = np.bincount(pq_dcodes_sub[is_sq], weights=wq[is_sq], minlength=n_drivers)
                    sqwval_sum = np.bincount(pq_dcodes_sub[is_sq], weights=wqval[is_sq], minlength=n_drivers)
                    valid_sq = sqw_sum > 0
                    sprint_q_form_index[valid_sq] = sqwval_sum[valid_sq] / np.maximum(sqw_sum[valid_sq], 1e-6)


        # Build samples for the current event
        evt_indices = np.where(evt_mask)[0]

        for idx in evt_indices:
            code = d_codes[idx]
            if not valid[code]:
                continue

            t_code = t_codes[idx]
            sample = {
                "driverId": uniques[code],
                "constructorId": t_uniques[t_code] if t_code >= 0 else None,
                "form_index": float(form_index[code]),
                "qualifying_form_index": float(q_form_index[code]),
                "team_form_index": float(team_form_index[t_code]) if t_code >= 0 else 0.0,
                "sprint_form_index": float(sprint_form_index[code]) if valid_sprint[code] else float(form_index[code]),
                "sprint_qualifying_form_index": float(sprint_q_form_index[code]) if valid_sq[code] else float(q_form_index[code]),
                "grid_finish_delta": float(gf_index[code]),
                "teammate_delta": float(tm_delta_index[code]),
                "circuit_avg_pos": float(c_avg_pos[code]),
                "circuit_dnf_rate": float(c_dnf_rate[code]),
                "circuit_experience": float(c_exp[code]),
                "grid": float(races_grid_float[idx]),
                "is_race": 1 if races_sessions[idx] == "race" else 0,
                "is_qualifying": 0,
                "is_sprint": 1 if races_sessions[idx] == "sprint" else 0,
            }
            rows.append(sample)

    if not rows:
        return None

    hist_df = pd.DataFrame(rows)

    # Ensure all numeric feature columns from X_current exist (fill missing with NaN)
    for col in X_current.columns:
        if col not in hist_df.columns and X_current[col].dtype in ("float64", "int64", "float32", "int32"):
            hist_df[col] = np.nan

    logger.info("[models] Built historical training set: %d samples from %d events",
                len(hist_df), len(event_keys))
    return hist_df


def _split_feature_columns(X: 'pd.DataFrame', exclude: List[str]) -> Tuple[List[str], List[str], List[str]]:
    """
    Returns (all_features, numeric_cols, categorical_cols), excluding provided columns.
    """
    features = [c for c in X.columns if c not in exclude]
    cat_cols = [c for c in features if X[c].dtype == "object"]
    num_cols = [c for c in features if c not in cat_cols]
    return features, num_cols, cat_cols


def train_pace_model(X: 'pd.DataFrame', session_type: str, cfg: Any = None,
                     hist_X: Optional['pd.DataFrame'] = None) -> Tuple[Any, 'np.ndarray', list]:
    """
    Train a model producing a "pace index" (lower is better/faster) per driver.

    The form_index is calculated as: -position + points, so HIGHER form_index = BETTER driver.
    For pace (where LOWER = FASTER), we use: y = -form_index as the target.

    The model learns to predict pace from features, then we blend with a baseline
    derived from form indices to ensure robust predictions even with limited features.

    Args:
        X: Feature matrix for the current event (inference target).
        session_type: Session type ('race', 'qualifying', 'sprint', etc.).
        cfg: Application configuration object.
        hist_X: Optional historical feature matrix to train on (out-of-sample).
            When provided, the GBM is trained on hist_X and predicted on X,
            preventing train/predict leakage.  When None, falls back to fitting
            on the current event X (legacy behaviour).
    """
    import numpy as np
    import pandas as pd
    from sklearn.preprocessing import StandardScaler, OneHotEncoder
    from sklearn.pipeline import Pipeline
    from sklearn.compose import ColumnTransformer
    from sklearn.impute import SimpleImputer
    from sklearn.ensemble import GradientBoostingRegressor

    # Determine training data: use historical data when available to avoid
    # fitting and predicting on the same event (train/predict leakage).
    if hist_X is not None and not hist_X.empty and "form_index" in hist_X.columns:
        X_train = hist_X
        logger.info("[models] Training GBM on %d historical samples (out-of-sample)", len(hist_X))
    else:
        X_train = X
        logger.info("[models] Training GBM on current event (%d rows, no historical X available)", len(X))


    # Target: use appropriate form index for the session type
    # form_index is HIGHER = BETTER, so negate for pace (LOWER = FASTER)
    if session_type == "sprint":
        target_col = "sprint_form_index"
    elif session_type == "sprint_qualifying":
        target_col = "sprint_qualifying_form_index"
    else:
        is_quali_session = session_type in ("qualifying", "sprint_qualifying")
        target_col = "qualifying_form_index" if is_quali_session else "form_index"

    if target_col not in X_train.columns:
        # Fallback to general form if specific form is missing
        if target_col == "sprint_form_index":
            target_col = "form_index"
        elif target_col == "sprint_qualifying_form_index":
            target_col = "qualifying_form_index"
            if target_col not in X_train.columns:
                target_col = "form_index"
        elif target_col == "qualifying_form_index":
            target_col = "form_index"
        else:
            target_col = "qualifying_form_index"


    if target_col not in X_train.columns:
        y = np.zeros(len(X_train), dtype=float)
    else:
        y = -X_train[target_col].astype(float).values

    # Features (exclude identifiers, session meta, and target to prevent leakage)
    # Note: we only exclude the current target column so the other index can be used as a feature
    exclude_cols = [
        "driverId", "name", "code", "constructorId", "constructorName", "number",
        "session_type", target_col, "current_quali_pos"
    ]
    
    # Also exclude columns that are entirely NaN in the training set
    all_nan_cols = [c for c in X_train.columns if c not in exclude_cols and X_train[c].isna().all()]
    # Keep columns that are in X (inference) even if they are all-NaN in history,
    # so they can show up in SHAP (with 0 influence if needed).
    cols_to_exclude = [c for c in all_nan_cols if c not in X.columns]
    exclude_cols.extend(cols_to_exclude)
    
    features, num_cols, cat_cols = _split_feature_columns(X_train, exclude=exclude_cols)
    logger.debug("[models] Final feature set (%d): %s", len(features), features)

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

    # Model selection with eager hyperparameters
    model = None

    # Try LightGBM
    try:
        import lightgbm as lgb
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
    except Exception:
        pass

    # Try XGBoost if LightGBM failed
    if model is None:
        try:
            import xgboost as xgb
            logger.warning("[models] LightGBM missing, falling back to XGBoost.")
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
        except Exception:
            pass

    # Fallback to sklearn
    if model is None:
        logger.warning("[models] LightGBM and XGBoost missing, falling back to Sklearn GradientBoostingRegressor.")
        model = GradientBoostingRegressor(
            n_estimators=100,
            learning_rate=0.1,
            max_depth=3,
            random_state=42
        )

    pipe = Pipeline(steps=[("pre", pre), ("model", model)])

    # Fit model on training data
    Xmat_train = X_train[features].copy()
    
    # Handle edge cases: 0 rows (no history) or 0 columns (no features found)
    if len(Xmat_train) == 0:
        yhat = np.zeros(len(X), dtype=float)
        # Dummy pipeline for shap compute
        from sklearn.dummy import DummyRegressor
        pipe = Pipeline(steps=[("pre", "passthrough"), ("model", DummyRegressor(strategy="constant", constant=0.0))])
        Xmat_train = pd.DataFrame({"const": np.zeros(1, dtype=float)})
        pipe.fit(Xmat_train, [0.0])
        features = ["const"]
    else:
        if Xmat_train.shape[1] == 0:
            Xmat_train = pd.DataFrame({"const": np.ones(len(X_train), dtype=float)})
            pipe = Pipeline(steps=[("pre", "passthrough"), ("model", model)])
            features = ["const"]

        pipe.fit(Xmat_train, y)

        # Predict on the current event (inference target), not the training data
        # Align inference columns to training features for consistency
        X_infer = X[features].copy() if set(features).issubset(X.columns) else X.reindex(columns=features)
        if features == ["const"]:
            X_infer = pd.DataFrame({"const": np.ones(len(X), dtype=float)})

        yhat = pipe.predict(X_infer)

    # Build baseline from form indices (for robustness)
    # Use appropriate form index as the baseline for this session type
    base = np.zeros(len(X), dtype=float)
    is_inference_quali = session_type in ("qualifying", "sprint_qualifying")
    infer_base_col = "qualifying_form_index" if is_inference_quali else "form_index"

    if infer_base_col not in X.columns:
         infer_base_col = "form_index" if infer_base_col == "qualifying_form_index" else "qualifying_form_index"

    if infer_base_col in X.columns:
        base = -X[infer_base_col].astype(float).values
    
    # Blending weights from config or defaults
    w_base_team = 0.3
    w_grid = 0.8
    w_gbm = 0.75
    w_base = 0.25
    w_quali = 0.5
    
    if cfg:
        try:
            w_gbm = cfg.modelling.blending.gbm_weight
            w_base = cfg.modelling.blending.baseline_weight
            w_base_team = cfg.modelling.blending.baseline_team_factor
            w_grid = getattr(cfg.modelling.blending, "grid_factor", 0.8)
            w_quali = getattr(cfg.modelling.blending, "current_quali_factor", 0.5)
        except AttributeError:
            pass

    if "team_form_index" in X.columns:
        # team_form_index is points-based, HIGHER = BETTER, so negate
        base = base - w_base_team * X["team_form_index"].astype(float).values

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
        # Model failed to learn, use baseline only
        pace_hat = base
    elif base_std < 1e-6 and yhat_std > 1e-6:
        # Baseline is flat (e.g., all same form), use model only
        pace_hat = yhat
    elif yhat_std > 1e-6 and base_std > 1e-6:
        # Both have signal - weight heavily towards trained model
        pace_hat = w_gbm * yhat + w_base * base
    else:
        # Both flat - use baseline with small noise for tie-breaking
        # ⚡ Bolt: standard_normal is slightly faster than normal(0, scale) due to less C overhead
        pace_hat = base + np.random.RandomState(42).standard_normal(len(base)) * 0.01

    # Stage 1.5: Blend current weekend qualifying position if available
    # This is a direct, high-weight signal from THIS weekend's qualifying
    # Applied before grid stickiness so both qualifying pace and grid influence the result
    if session_type in ("race", "sprint") and "current_quali_pos" in X.columns:
        quali_vals = X["current_quali_pos"].astype(float).values
        has_quali = ~np.isnan(quali_vals)
        if has_quali.any():
            # Normalize pace_hat to z-score for fair blending
            p_mu = float(np.nanmean(pace_hat))
            p_sd = float(np.nanstd(pace_hat))
            if p_sd > 1e-6:
                pace_z = (pace_hat - p_mu) / p_sd
            else:
                pace_z = pace_hat - p_mu

            # Normalize qualifying positions to z-score (lower position = faster = lower z)
            q_mu = float(np.nanmean(quali_vals[has_quali]))
            q_sd = float(np.nanstd(quali_vals[has_quali]))
            if q_sd > 1e-6:
                quali_z = (quali_vals - q_mu) / q_sd
            else:
                quali_z = quali_vals - q_mu

            # Blend: w_quali controls how much current qualifying overrides historical form
            blended_z = np.where(
                has_quali,
                (1.0 - w_quali) * pace_z + w_quali * quali_z,
                pace_z  # Keep original for drivers without qualifying data
            )

            # Rescale back to original pace scale
            if p_sd > 1e-6:
                pace_hat = blended_z * p_sd + p_mu
            else:
                pace_hat = blended_z + p_mu

            logger.info(
                "[models] Blended current_quali_pos (factor=%.2f) for %d/%d drivers",
                w_quali, has_quali.sum(), len(X)
            )

    # Second Stage: Incorporate grid "stickiness" for race sessions
    # This ensures starting position acts as a strong anchor for the prediction.
    if session_type in ("race", "sprint") and "grid" in X.columns:
        # Dynamic stickiness based on circuit passability and driver skill.
        # The base w_grid is calibrated; circuit and driver adjustments scale
        # proportionally to the flexibility budget (1 - w_grid) so they
        # automatically adapt when the calibrated grid factor changes.
        dynamic_w_grid = w_grid
        flexibility = max(1.0 - w_grid, 0.05)  # how much non-grid signal is allowed

        # 1. Circuit overtake difficulty
        if "circuit_overtake_difficulty" in X.columns:
            avg_circuit_diff = float(np.nanmean(X["circuit_overtake_difficulty"]))
            if not np.isnan(avg_circuit_diff):
                # Negative = hard to pass -> increase stickiness
                circuit_modifier = -avg_circuit_diff * flexibility * 0.15
                dynamic_w_grid = dynamic_w_grid + circuit_modifier

        # 2. Driver overtake propensity
        if "grid_finish_delta" in X.columns:
            # Positive = driver gains positions -> decrease stickiness
            driver_gfd = X["grid_finish_delta"].astype(float).values
            driver_modifier = -np.nan_to_num(driver_gfd) * flexibility * 0.08
            dynamic_w_grid = dynamic_w_grid + driver_modifier

        dynamic_w_grid = np.clip(dynamic_w_grid, 0.4, 0.95)

        # 1. Grid is the absolute anchor
        # Use actual grid if available, otherwise fallback to index/average
        grid_fallback = float(len(X)) if len(X) > 0 else 20.0
        grid_vals = X["grid"].astype(float).fillna(grid_fallback).values
        
        # 2. Normalize pace_hat to determine pace relative advantage (z-score)
        mu = float(np.nanmean(pace_hat))
        sd = float(np.nanstd(pace_hat))
        if sd > 1e-6:
            pace_z = (pace_hat - mu) / sd
        else:
            pace_z = pace_hat - mu
            
        # 3. Apply pace advantage as a Delta to the anchor
        # Max reasonable delta bounded to 10 positions (a 2 sigma pace advantage = ~5 grid slots)
        MAX_DELTA = 10.0
        pace_multiplier = 1.0 - dynamic_w_grid
        pace_delta = pace_z * pace_multiplier * MAX_DELTA
        
        # 4. Final simulated baseline is Grid + Expected Delta
        pace_hat = grid_vals + pace_delta
        
        logger.info(
            "[models] Applied dynamic Anchor-Delta grid stickiness modified by circuit and driver"
        )
        
        # DEBUG PRINTS FOR UNDERSTANDING
        if "driverId" in X.columns and session_type == "race":
            for i, drv in enumerate(X["driverId"]):
                if drv in ("antonelli", "hadjar", "piastri"):
                    logger.info(f"[DEBUG] {drv} | Grid: {grid_vals[i]} | PaceZ: {pace_z[i]:.2f} | Delta: {pace_delta[i]:.2f} | Final PaceHat: {pace_hat[i]:.2f}")

    # Compute SHAP values for explainability
    shap_per_driver = None
    try:
        shap_per_driver = compute_shap_values(pipe, X, features)
    except Exception as e:
        logger.debug("[models] SHAP computation failed (non-fatal): %s", e)

    return pipe, pace_hat, features, shap_per_driver


def compute_shap_values(
    pipe: Any,
    X: 'pd.DataFrame',
    features: List[str],
) -> Optional[List[Dict[str, float]]]:
    """Compute per-driver SHAP values for the GBM component.

    Returns a list of dicts (one per driver row in X) mapping original feature
    names to their SHAP contribution (lower contribution = made prediction worse,
    i.e. slower pace; the sign indicates direction of influence on the pace index).

    Returns None if shap is unavailable or the model type is not supported.
    """

    try:
        import shap as shap_lib
    except ImportError:
        logger.debug("[models] shap library not installed; using fallback feature contributions")
        return _fallback_feature_contributions(pipe, X, features)

    try:
        model = pipe.named_steps["model"]
        pre = pipe.named_steps["pre"]

        # Build transformed feature matrix (same as what the model saw at training)
        X_infer = X[features].copy() if set(features).issubset(X.columns) else X.reindex(columns=features)
        X_transformed = pre.transform(X_infer)

        # Resolve output feature names after one-hot encoding
        try:
            transformed_names = pre.get_feature_names_out()
        except Exception:
            transformed_names = None

        # Build explainer — TreeExplainer works natively for LightGBM, XGBoost,
        # and sklearn GBM without needing a background dataset.
        explainer = shap_lib.TreeExplainer(model)
        # Using check_additivity=False to prevent additivity check failures which are common with LightGBM/XGBoost
        try:
            shap_values = explainer.shap_values(X_transformed, check_additivity=False)
        except Exception:
            shap_values = explainer.shap_values(X_transformed)

        # Extract raw values from Explanation objects (newer shap versions)
        if hasattr(shap_values, "values"):
            shap_values = shap_values.values

        # Scikit-learn's GradientBoostingRegressor can sometimes return a single-item list of arrays
        if isinstance(shap_values, list) and len(shap_values) > 0:
            shap_values = shap_values[0]

        # shap_values shape: (n_drivers, n_transformed_features)
        if shap_values is None or len(shap_values) == 0:
            return _fallback_feature_contributions(pipe, X, features)

        # Map transformed feature names back to original feature names.
        # For numeric features the mapping is 1-to-1.
        # For one-hot encoded categoricals (e.g. constructorId_ferrari) we attribute
        # the SHAP contribution back to the original column name by summing
        # across all dummy columns for that original feature.
        n_drivers = len(X)
        result: List[Dict[str, float]] = []

        for i in range(n_drivers):
            row_shap = shap_values[i]  # shape: (n_transformed_features,)
            contrib: Dict[str, float] = {}

            if transformed_names is not None and len(transformed_names) == len(row_shap):
                for j, tname in enumerate(transformed_names):
                    # sklearn ColumnTransformer prefixes names like "num__form_index"
                    # or "cat__constructorId_ferrari"
                    if "__" in tname:
                        prefix, rest = tname.split("__", 1)
                        # For OHE columns, rest is "originalcol_value"; extract original col
                        # by matching against known feature names
                        orig_col = None
                        for feat in features:
                            if rest == feat or rest.startswith(feat + "_"):
                                orig_col = feat
                                break
                        if orig_col is None:
                            orig_col = rest
                    else:
                        orig_col = tname

                    contrib[orig_col] = contrib.get(orig_col, 0.0) + float(row_shap[j])
            else:
                # Fallback: match by position to original numeric features
                for j, fname in enumerate(features):
                    if j < len(row_shap):
                        contrib[fname] = float(row_shap[j])

            result.append(contrib)

        logger.info("[models] SHAP values computed for %d drivers", n_drivers)
        return result

    except Exception as e:
        logger.debug("[models] SHAP computation error: %s; using fallback feature contributions", e)
        return _fallback_feature_contributions(pipe, X, features)


def _fallback_feature_contributions(
    pipe: Any,
    X: 'pd.DataFrame',
    features: List[str],
) -> Optional[List[Dict[str, float]]]:
    """Fallback per-driver feature contributions when SHAP is unavailable.

    Uses transformed feature values scaled by model feature importance.
    This is not a true Shapley attribution but provides a stable per-driver,
    per-feature directional influence estimate for UI display.
    """
    try:
        import numpy as np

        model = pipe.named_steps["model"]
        pre = pipe.named_steps["pre"]

        X_infer = X[features].copy() if set(features).issubset(X.columns) else X.reindex(columns=features)
        X_transformed = pre.transform(X_infer)

        try:
            transformed_names = list(pre.get_feature_names_out())
        except Exception:
            transformed_names = [f"f{i}" for i in range(X_transformed.shape[1])]

        importances = getattr(model, "feature_importances_", None)
        if importances is None:
            coef = getattr(model, "coef_", None)
            if coef is not None:
                importances = np.abs(np.asarray(coef)).ravel()

        if importances is None or len(importances) != X_transformed.shape[1]:
            importances = np.ones(X_transformed.shape[1], dtype=float)

        weighted = np.asarray(X_transformed, dtype=float) * np.asarray(importances, dtype=float)

        result: List[Dict[str, float]] = []
        for i in range(weighted.shape[0]):
            contrib: Dict[str, float] = {}
            for j, tname in enumerate(transformed_names):
                if "__" in str(tname):
                    _, rest = str(tname).split("__", 1)
                    orig_col = None
                    for feat in features:
                        if rest == feat or rest.startswith(feat + "_"):
                            orig_col = feat
                            break
                    if orig_col is None:
                        orig_col = rest
                else:
                    orig_col = str(tname)

                contrib[orig_col] = contrib.get(orig_col, 0.0) + float(weighted[i, j])
            result.append(contrib)

        return result
    except Exception as e:
        logger.debug("[models] Fallback feature contribution error: %s", e)
        return None


def estimate_dnf_probabilities(
    hist: 'pd.DataFrame',
    current_X: 'pd.DataFrame',
    alpha: float = 2.0,
    beta: float = 8.0,
    driver_weight: float = 0.6,
    team_weight: float = 0.4,
    clip_min: float = 0.02,
    clip_max: float = 0.30,
    cfg: Any = None,
    event_weather: Optional[Dict[str, float]] = None,
    hist_weather: Optional[Dict[Tuple[int, int], Dict[str, float]]] = None,
) -> 'np.ndarray':
    """
    Estimate per-driver DNF probabilities using Beta-smoothed empirical base rates.
    
    Weather-aware: calculates separate DNF rates for wet vs dry conditions and uses
    the appropriate category based on the forecast for the current session.
    Falls back to overall rates when weather-specific data is insufficient.
    
    Arguments allow overrides, but cfg takes precedence if provided.
    """
    import numpy as np
    import pandas as pd

    if cfg:
        try:
            alpha = cfg.modelling.dnf.alpha
            beta = cfg.modelling.dnf.beta
            driver_weight = cfg.modelling.dnf.driver_weight
            team_weight = cfg.modelling.dnf.team_weight
            clip_min = cfg.modelling.dnf.clip_min
            clip_max = cfg.modelling.dnf.clip_max
        except AttributeError:
            pass

    races = hist[hist["session"] == "race"].copy()
    if races.empty or current_X is None or current_X.empty:
        return np.full(len(current_X) if current_X is not None else 0, 0.08, dtype=float)

    # Detect DNF status
    if "is_dnf" in races.columns:
        races["dnf"] = races["is_dnf"].astype(int)
    else:
        status = races["status"].astype(str).str.lower()
        dnf = (~races["position"].notna()) | status.str.contains(
            "accident|engine|gear|suspension|electrical|hydraulics|dnf|brake|clutch|collision|spin|damage"
        )
        races["dnf"] = dnf.astype(int)

    # Determine if current session is wet (rain > 1mm threshold)
    WET_THRESHOLD = 1.0  # mm of rain
    current_is_wet = False
    if event_weather:
        rain_sum = event_weather.get("rain_sum", 0.0)
        if rain_sum is not None and not (rain_sum != rain_sum):  # NaN check
            current_is_wet = float(rain_sum) > WET_THRESHOLD

    # Tag historical races as wet/dry if historical weather is available
    races["is_wet"] = False
    if hist_weather:
        # Optimization: Vectorized weather lookup via merge (~50x faster)
        # Handle potential bad data safely (original loop had try-except)

        # 1. Prepare weather reference DataFrame
        w_records = []
        for (s, r), w in hist_weather.items():
            rain = w.get("rain_sum", 0.0)
            if rain is not None and not (rain != rain):  # NaN check
                w_records.append({"season": int(s), "round": int(r), "rain_sum": float(rain)})

        if w_records:
            w_df = pd.DataFrame(w_records)

            # 2. Prepare join keys on races safely
            # Convert to numeric, coercing errors to NaN, then fill with -1 (safe sentinel)
            races["_m_season"] = pd.to_numeric(races["season"], errors="coerce").fillna(-1).astype(int)
            races["_m_round"] = pd.to_numeric(races["round"], errors="coerce").fillna(-1).astype(int)

            # 3. Vectorized merge
            # Note: races is a local copy, so rebinding it via merge is safe
            races = races.merge(
                w_df,
                left_on=["_m_season", "_m_round"],
                right_on=["season", "round"],
                how="left",
                suffixes=("", "_w")
            )

            # 4. Apply threshold
            # missing weather (NaN) -> 0.0 -> False
            races["is_wet"] = races["rain_sum"].fillna(0.0) > WET_THRESHOLD

            # 5. Cleanup
            races.drop(columns=["_m_season", "_m_round", "rain_sum", "season_w", "round_w"], inplace=True, errors="ignore")

    # Ensure fallback even if current is totally missing
    weather_races = races[races["is_wet"] == current_is_wet]
    N_weather = len(weather_races)
    
    # Smooth blend weight: 0 races -> 0.0. 15+ races -> 1.0
    weather_weight = min(1.0, N_weather / 15.0)

    # Calculate overall rates
    drv_counts = races.groupby("driverId")["dnf"].agg(["sum", "count"]).rename(columns={"sum": "k", "count": "n"})
    drv_p_all = (drv_counts["k"] + alpha) / (drv_counts["n"] + alpha + beta)
    
    team_counts = races.groupby("constructorId")["dnf"].agg(["sum", "count"]).rename(columns={"sum": "k", "count": "n"})
    team_p_all = (team_counts["k"] + alpha) / (team_counts["n"] + alpha + beta)

    # Calculate weather specific rates
    w_drv_counts = weather_races.groupby("driverId")["dnf"].agg(["sum", "count"]).rename(columns={"sum": "k", "count": "n"})
    drv_p_w = (w_drv_counts["k"] + alpha) / (w_drv_counts["n"] + alpha + beta)
    
    w_team_counts = weather_races.groupby("constructorId")["dnf"].agg(["sum", "count"]).rename(columns={"sum": "k", "count": "n"})
    team_p_w = (w_team_counts["k"] + alpha) / (w_team_counts["n"] + alpha + beta)

    # Blend them
    drv_p_blended = weather_weight * drv_p_w.reindex(drv_p_all.index).fillna(drv_p_all) + (1.0 - weather_weight) * drv_p_all
    team_p_blended = weather_weight * team_p_w.reindex(team_p_all.index).fillna(team_p_all) + (1.0 - weather_weight) * team_p_all

    # Global fallback rate
    global_k = races["dnf"].sum()
    global_n = races.shape[0]
    global_p = (global_k + alpha) / (global_n + alpha + beta)
    
    # Circuit multiplier
    circuit_dnf = 0.08
    if current_X is not None and "global_circuit_dnf_rate" in current_X.columns:
        c_mean = current_X["global_circuit_dnf_rate"].mean()
        if pd.notna(c_mean):
            circuit_dnf = float(c_mean)
    circuit_modifier = circuit_dnf / 0.08

    drv_map = drv_p_blended.to_dict()
    team_map = team_p_blended.to_dict()

    p_drv = current_X["driverId"].map(drv_map).astype(float).fillna(global_p)
    p_team = current_X["constructorId"].map(team_map).astype(float).fillna(global_p)

    p = driver_weight * p_drv.values + team_weight * p_team.values
    p = p * circuit_modifier
    p = np.clip(p.astype(float), clip_min, clip_max)
    return p
