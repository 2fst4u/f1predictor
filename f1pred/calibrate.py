"""Dynamic calibration of ALL model parameters based on recent history.

This module provides the CalibrationManager to optimize every tunable weight
and hyperparameter by minimizing prediction error over a sliding lookback
window using per-event Spearman rank correlation as the primary objective.

All modelling weights are calibrated at runtime — config.yaml values serve
only as initial defaults for the first run.  After calibration completes,
the authoritative values live in calibration_weights.json.

Calibrated parameter groups:
    - blending: GBM/baseline weights, team/driver-team factors, grid stickiness,
                season weights, qualifying factor, analytical win weight
    - ensemble: w_gbm, w_elo, w_bt, w_mixed
    - dnf:      alpha, beta, driver_weight, team_weight
    - simulation: noise_factor, min_noise
    - recency:  half_life_base, half_life_team
    - elo:      k (Elo update strength)
"""
from __future__ import annotations

import json
import time
from pathlib import Path
from typing import Dict, Any, Optional, Tuple, TYPE_CHECKING
from datetime import datetime, timedelta, timezone
from concurrent.futures import ThreadPoolExecutor

from .util import get_logger, ensure_dirs
from .data.jolpica import JolpicaClient
from .data.open_meteo import OpenMeteoClient
from .ensemble import EnsembleConfig

if TYPE_CHECKING:
    import pandas as pd

logger = get_logger(__name__)

# ---------------------------------------------------------------------------
# Parameter vector layout
# ---------------------------------------------------------------------------
# Indices into the flat optimisation vector and their human-readable names.
# Changing the order here requires updating _pack / _unpack helpers below.

PARAM_NAMES = [
    # -- blending (0-4) --
    "gbm_weight",               # 0
    "baseline_weight",          # 1
    "baseline_team_factor",     # 2
    "baseline_driver_team_factor",  # 3
    "grid_factor",              # 4
    # -- ensemble (5-8) --
    "ens_pace",                 # 5
    "ens_elo",                  # 6
    "ens_bt",                   # 7
    "ens_mixed",                # 8
    # -- season / quali (9-12) --
    "current_season_weight",    # 9
    "current_season_qualifying_weight",  # 10
    "current_quali_factor",     # 11
    "analytical_win_weight",    # 12
    # -- DNF (13-16) --
    "dnf_alpha",                # 13
    "dnf_beta",                 # 14
    "dnf_driver_weight",        # 15
    "dnf_team_weight",          # 16
    # -- simulation (17-18) --
    "noise_factor",             # 17
    "min_noise",                # 18
    # -- recency (19-20) --
    "half_life_base",           # 19
    "half_life_team",           # 20
    # -- elo (21) --
    "elo_k",                    # 21
]

N_PARAMS = len(PARAM_NAMES)

# Bounds for L-BFGS-B  (lower, upper) per parameter
PARAM_BOUNDS = [
    (0.05, 1.0),    # 0  gbm_weight
    (0.05, 1.0),    # 1  baseline_weight
    (0.2, 2.0),     # 2  baseline_team_factor
    (0.0, 2.0),     # 3  baseline_driver_team_factor
    (0.1, 0.95),    # 4  grid_factor
    (0.0, 1.0),     # 5  ens_pace
    (0.0, 1.0),     # 6  ens_elo
    (0.0, 1.0),     # 7  ens_bt
    (0.0, 1.0),     # 8  ens_mixed
    (3.0, 50.0),    # 9  current_season_weight
    (3.0, 50.0),    # 10 current_season_qualifying_weight
    (0.0, 1.0),     # 11 current_quali_factor
    (0.0, 1.0),     # 12 analytical_win_weight
    (0.1, 10.0),    # 13 dnf_alpha
    (0.1, 30.0),    # 14 dnf_beta
    (0.0, 1.0),     # 15 dnf_driver_weight
    (0.0, 1.0),     # 16 dnf_team_weight
    (0.01, 0.5),    # 17 noise_factor
    (0.01, 0.3),    # 18 min_noise
    (30.0, 500.0),  # 19 half_life_base
    (60.0, 800.0),  # 20 half_life_team
    (5.0, 60.0),    # 21 elo_k
]

# Default / initial guess (matches config.yaml defaults)
PARAM_DEFAULTS = [
    0.75,   # 0  gbm_weight
    0.25,   # 1  baseline_weight
    0.3,    # 2  baseline_team_factor
    0.2,    # 3  baseline_driver_team_factor
    0.8,    # 4  grid_factor
    0.4,    # 5  ens_pace
    0.2,    # 6  ens_elo
    0.2,    # 7  ens_bt
    0.2,    # 8  ens_mixed
    8.0,    # 9  current_season_weight
    8.0,    # 10 current_season_qualifying_weight
    0.5,    # 11 current_quali_factor
    0.5,    # 12 analytical_win_weight
    2.0,    # 13 dnf_alpha
    8.0,    # 14 dnf_beta
    0.6,    # 15 dnf_driver_weight
    0.4,    # 16 dnf_team_weight
    0.15,   # 17 noise_factor
    0.05,   # 18 min_noise
    120.0,  # 19 half_life_base
    240.0,  # 20 half_life_team
    20.0,   # 21 elo_k
]


def _calibration_version() -> str:
    """Derive a version fingerprint from the calibration schema.

    Changes to the parameter layout (names, bounds, defaults, count)
    produce a different hash, which forces a fresh calibration when
    the application is updated.  This avoids silently reusing weights
    produced by an older objective function or parameter layout.
    """
    import hashlib
    sig = (
        str(PARAM_NAMES)
        + str(PARAM_BOUNDS)
        + str(PARAM_DEFAULTS)
        + str(N_PARAMS)
    )
    return hashlib.sha256(sig.encode()).hexdigest()[:12]


CALIBRATION_VERSION = _calibration_version()


def _unpack_weights(w) -> Dict[str, Any]:
    """Convert flat parameter vector to nested weights dict."""
    import numpy as np
    w = np.asarray(w, dtype=float)

    # Normalise ensemble weights to sum to 1
    ens_raw = w[5:9].copy()
    ens_sum = ens_raw.sum()
    if ens_sum > 0:
        ens_raw /= ens_sum
    else:
        ens_raw[:] = 0.25

    # Normalise blending gbm/baseline to partition of unity
    gbm_raw = max(1e-3, w[0])
    base_raw = max(1e-3, w[1])
    total_main = gbm_raw + base_raw

    return {
        "ensemble": {
            "w_gbm": float(ens_raw[0]),
            "w_elo": float(ens_raw[1]),
            "w_bt": float(ens_raw[2]),
            "w_mixed": float(ens_raw[3]),
        },
        "blending": {
            "gbm_weight": float(gbm_raw / total_main),
            "baseline_weight": float(base_raw / total_main),
            "baseline_team_factor": float(w[2]),
            "baseline_driver_team_factor": float(w[3]),
            "grid_factor": float(np.clip(w[4], 0.0, 1.0)),
            "current_season_weight": float(w[9]),
            "current_season_qualifying_weight": float(w[10]),
            "current_quali_factor": float(np.clip(w[11], 0.0, 1.0)),
            "analytical_win_weight": float(np.clip(w[12], 0.0, 1.0)),
        },
        "dnf": {
            "alpha": float(max(0.1, w[13])),
            "beta": float(max(0.1, w[14])),
            "driver_weight": float(np.clip(w[15], 0.0, 1.0)),
            "team_weight": float(np.clip(w[16], 0.0, 1.0)),
        },
        "simulation": {
            "noise_factor": float(max(0.01, w[17])),
            "min_noise": float(max(0.01, w[18])),
        },
        "recency": {
            "half_life_base": float(max(30.0, w[19])),
            "half_life_team": float(max(60.0, w[20])),
        },
        "elo": {
            "k": float(np.clip(w[21], 5.0, 60.0)),
        },
    }


def _pack_weights(d: Dict[str, Any]) -> list:
    """Convert nested weights dict back to flat parameter vector."""
    b = d.get("blending", {})
    e = d.get("ensemble", {})
    dn = d.get("dnf", {})
    sim = d.get("simulation", {})
    rec = d.get("recency", {})
    elo = d.get("elo", {})

    return [
        b.get("gbm_weight", PARAM_DEFAULTS[0]),
        b.get("baseline_weight", PARAM_DEFAULTS[1]),
        b.get("baseline_team_factor", PARAM_DEFAULTS[2]),
        b.get("baseline_driver_team_factor", PARAM_DEFAULTS[3]),
        b.get("grid_factor", PARAM_DEFAULTS[4]),
        e.get("w_gbm", PARAM_DEFAULTS[5]),
        e.get("w_elo", PARAM_DEFAULTS[6]),
        e.get("w_bt", PARAM_DEFAULTS[7]),
        e.get("w_mixed", PARAM_DEFAULTS[8]),
        b.get("current_season_weight", PARAM_DEFAULTS[9]),
        b.get("current_season_qualifying_weight", PARAM_DEFAULTS[10]),
        b.get("current_quali_factor", PARAM_DEFAULTS[11]),
        b.get("analytical_win_weight", PARAM_DEFAULTS[12]),
        dn.get("alpha", PARAM_DEFAULTS[13]),
        dn.get("beta", PARAM_DEFAULTS[14]),
        dn.get("driver_weight", PARAM_DEFAULTS[15]),
        dn.get("team_weight", PARAM_DEFAULTS[16]),
        sim.get("noise_factor", PARAM_DEFAULTS[17]),
        sim.get("min_noise", PARAM_DEFAULTS[18]),
        rec.get("half_life_base", PARAM_DEFAULTS[19]),
        rec.get("half_life_team", PARAM_DEFAULTS[20]),
        elo.get("k", PARAM_DEFAULTS[21]),
    ]


class CalibrationManager:
    def __init__(self, cfg):
        self.cfg = cfg
        # Default path relative to project or absolute if configured
        self.weights_file = Path(cfg.calibration.weights_file)
        self.lookback_days = getattr(cfg.calibration, "lookback_window_days", 1095)
        self.frequency_hours = getattr(cfg.calibration, "frequency_hours", 24)

        # Default weights (fallback) — uses the full expanded structure
        self.current_weights = _unpack_weights(PARAM_DEFAULTS)
        self.last_race_id: Optional[str] = None

    def load_weights(self) -> Dict[str, Any]:
        """Load weights from disk if available, otherwise return defaults."""
        if self.weights_file.exists():
            try:
                with open(self.weights_file, "r") as f:
                    data = json.load(f)

                # Validate: must have at least ensemble + blending
                if "ensemble" in data and "blending" in data:
                    self.current_weights = data
                    self.last_race_id = data.get("last_race_id")
                    logger.info(
                        "[calibrate] Loaded calibrated weights from %s (last_race_id=%s)",
                        self.weights_file, self.last_race_id,
                    )
                else:
                    logger.warning("[calibrate] Weights file malformed, using defaults")
            except Exception as e:
                logger.error("[calibrate] Failed to load weights: %s", e)

        return self.current_weights

    def save_weights(self):
        """Save current weights to disk."""
        try:
            ensure_dirs(str(self.weights_file.parent))
            with open(self.weights_file, "w") as f:
                data = self.current_weights.copy()
                if self.last_race_id:
                    data["last_race_id"] = self.last_race_id
                data["calibration_version"] = CALIBRATION_VERSION
                data["calibration_timestamp"] = datetime.now(timezone.utc).isoformat()
                json.dump(data, f, indent=2)
            logger.info("[calibrate] Saved calibrated weights to %s", self.weights_file)
        except Exception as e:
            logger.error("[calibrate] Failed to save weights: %s", e)

    def check_calibration_needed(self, history_df: Optional['pd.DataFrame'] = None) -> bool:  # noqa: F821
        """Check if calibration is needed based on new race results or missing file."""
        if not self.cfg.calibration.enabled:
            return False

        if not self.weights_file.exists():
            logger.info("[calibrate] Weights file missing, calibration needed")
            return True

        # Load existing weights (and last_race_id)
        self.load_weights()

        # Version check: if the calibration schema has changed (parameters,
        # bounds, defaults), the cached weights are stale and must be rebuilt.
        stored_version = self.current_weights.get("calibration_version")
        if stored_version != CALIBRATION_VERSION:
            logger.info(
                "[calibrate] Calibration version mismatch (stored=%s, current=%s), recalibration needed",
                stored_version, CALIBRATION_VERSION,
            )
            return True

        # Smart Check: Has a new race occurred?
        if history_df is not None and not history_df.empty:
            # Find the latest completed race in the history
            latest_race = history_df[history_df["session"] == "race"].sort_values("date").iloc[-1]
            latest_id = f"{latest_race['season']}_{latest_race['round']}"

            if self.last_race_id != latest_id:
                logger.info(
                    "[calibrate] New race result found (%s != %s), calibration needed",
                    latest_id, self.last_race_id,
                )
                return True
            else:
                logger.debug("[calibrate] No new results since %s. Skipping calibration.", self.last_race_id)
                return False

        # Fallback to time-based check if history not provided
        try:
            mtime = datetime.fromtimestamp(self.weights_file.stat().st_mtime, tz=timezone.utc)
            age = datetime.now(timezone.utc) - mtime
            age_hours = age.total_seconds() / 3600.0

            if age_hours > self.frequency_hours:
                logger.info(
                    "[calibrate] Weights file is %.1fh old (> %dh), calibration needed",
                    age_hours, self.frequency_hours,
                )
                return True
        except Exception:
            return True

        return False

    # ------------------------------------------------------------------
    # Full calibration
    # ------------------------------------------------------------------
    def run_calibration(self, jc: JolpicaClient, om: OpenMeteoClient,
                        history_df: Optional['pd.DataFrame'] = None):  # noqa: F821
        """Run the full calibration process over all tunable parameters."""
        # Import heavy deps here
        import numpy as np
        import pandas as pd
        from scipy.optimize import minimize
        from scipy.stats import spearmanr
        from .features import build_session_features, collect_historical_results
        from .models import train_pace_model
        from .ensemble import EloModel, BradleyTerryModel, MixedEffectsLikeModel

        logger.info("[calibrate] Starting full calibration (%d parameters)...", N_PARAMS)
        if history_df is None or history_df.empty:
            print("    [Calibration] Starting self-training process... (fetching history)", flush=True)
        else:
            print("    [Calibration] Starting self-training process... (using pre-fetched history)", flush=True)

        t0 = time.time()

        try:
            # 1. Define window
            now = datetime.now(timezone.utc)
            start_date = now - timedelta(days=self.lookback_days)

            # 2. Fetch History
            if history_df is None or history_df.empty:
                logger.info("[calibrate] Fetching history...")
                print("    [Calibration] Fetching historical race data (last 4 years)...", flush=True)
                try:
                    all_hist = collect_historical_results(
                        jc,
                        season=now.year,
                        end_before=now,
                        lookback_years=4,
                    )
                except Exception as e:
                    logger.error("[calibrate] History fetch failed: %s", e)
                    print(f"    [Calibration] Error fetching history: {e}")
                    return
            else:
                all_hist = history_df
                all_hist = all_hist[all_hist["date"] < now]

            # Filter specifically for the calibration set (races in the lookback window)
            calib_races = all_hist[
                (all_hist["session"] == "race")
                & (all_hist["date"] >= start_date)
                & (all_hist["position"].notna())
            ].sort_values("date")

            if calib_races.empty:
                logger.warning("[calibrate] No races found in lookback window. Skipping.")
                print("    [Calibration] SKIPPED: No recent races found in lookback window.")
                return

            # Identify unique events
            events = calib_races[["season", "round", "date"]].drop_duplicates().sort_values("date")
            logger.info(
                "[calibrate] Found %d events in lookback window (%s to %s)",
                len(events), start_date.date(), now.date(),
            )

            # Update last_race_id
            if not events.empty:
                last_evt = events.iloc[-1]
                self.last_race_id = f"{last_evt['season']}_{last_evt['round']}"

            # 3. Out-of-time training split
            train_cutoff = start_date
            train_hist = all_hist[all_hist["date"] < train_cutoff]

            if train_hist.empty:
                logger.warning("[calibrate] No training history prior to lookback window. Using 70/30 split.")
                split_idx = int(len(all_hist) * 0.7)
                all_hist_sorted = all_hist.sort_values("date")
                train_hist = all_hist_sorted.iloc[:split_idx]
                calib_races = all_hist_sorted.iloc[split_idx:]
                calib_races = calib_races[calib_races["session"] == "race"]
                events = calib_races[["season", "round", "date"]].drop_duplicates().sort_values("date")

            # 4. Train GBM Baseline on pre-window data
            logger.info("[calibrate] Training baseline GBM on %d rows (pre-%s)...",
                        len(train_hist), train_cutoff.date())

            train_race_hist = train_hist[train_hist["session"] == "race"].tail(500)
            top_train_events = train_race_hist[["season", "round", "date"]].drop_duplicates().to_dict("records")

            X_train_list = []
            with ThreadPoolExecutor(max_workers=4) as executor:
                futures = []
                for evt in top_train_events:
                    futures.append(executor.submit(
                        build_session_features,
                        jc, om,
                        int(evt["season"]), int(evt["round"]), "race",
                        pd.Timestamp(evt["date"]), self.cfg,
                    ))
                for f in futures:
                    try:
                        X_i, _, _ = f.result()
                        if X_i is not None and not X_i.empty:
                            X_train_list.append(X_i)
                    except Exception as e:
                        logger.error("[calibrate] Training sample generation failed: %s", e)

            if not X_train_list:
                logger.error("[calibrate] Failed to generate training data. Aborting.")
                print("    [Calibration] ABORTED: Failed to generate training data.")
                return

            X_train = pd.concat(X_train_list, ignore_index=True)
            logger.info("[calibrate] Generated %d training samples.", len(X_train))

            gbm_pipe, _, gbm_features = train_pace_model(X_train, "race", self.cfg)

            # 5. Collect calibration data from each event
            #
            # To allow the optimizer to learn current_season_weight, we store
            # BOTH the boosted form indices (used by blending params 0-4) AND
            # separate unboosted vs current-season-only form components so the
            # objective can recompute an effective form at different boost levels.
            calibration_data = []

            logger.info("[calibrate] Generating scores for calibration events...")
            for i, evt in enumerate(events.itertuples()):
                s, r, d = int(evt.season), int(evt.round), pd.Timestamp(evt.date)

                X_evt, _, roster = build_session_features(jc, om, s, r, "race", d, self.cfg)
                if X_evt is None or X_evt.empty:
                    continue

                # GBM Score — align columns to match training features
                try:
                    X_pred = X_evt.reindex(columns=gbm_features) if gbm_features else X_evt
                    pace_hat = gbm_pipe.predict(X_pred)
                    pace_hat = (pace_hat - pace_hat.mean()) / (pace_hat.std() + 1e-6)
                except Exception as e:
                    logger.warning("[calibrate] GBM predict failed for %d R%d: %s", s, r, e)
                    continue

                # History up to this race (strict temporal cutoff)
                hist_subset = all_hist[all_hist["date"] < d]

                # Ensemble scores
                elo = EloModel().fit(hist_subset).predict(X_evt)
                bt = BradleyTerryModel().fit(hist_subset).predict(X_evt)
                mixed = MixedEffectsLikeModel().fit(hist_subset).predict(X_evt)

                # Baseline components (boosted form as computed by features.py)
                base_form = -X_evt["form_index"].fillna(0).astype(float).values
                base_team = -X_evt.get("team_form_index", pd.Series(0, index=X_evt.index)).fillna(0).astype(float).values
                base_dt = -X_evt.get("driver_team_form_index", pd.Series(0, index=X_evt.index)).fillna(0).astype(float).values

                # Compute UNBOOSTED form indices for the same drivers so the
                # objective function can recompute form = unboosted + boost * cur_season_only.
                # This gives the optimizer a gradient signal on current_season_weight.
                try:
                    from .features import compute_form_indices
                    hl_base = self.cfg.modelling.recency_half_life_days.base

                    # Unboosted form (boost_factor=1.0 => no current-season emphasis)
                    unboosted_form_df = compute_form_indices(
                        hist_subset, ref_date=d, half_life_days=hl_base,
                        current_season=None, boost_factor=1.0,
                    )
                    unboosted_map = dict(zip(unboosted_form_df["driverId"], -unboosted_form_df["form_index"]))

                    # Current-season-only form (only races in the event's season)
                    cur_season_hist = hist_subset[hist_subset["season"] == s]
                    if not cur_season_hist.empty:
                        cur_form_df = compute_form_indices(
                            cur_season_hist, ref_date=d, half_life_days=hl_base,
                            current_season=None, boost_factor=1.0,
                        )
                        cur_form_map = dict(zip(cur_form_df["driverId"], -cur_form_df["form_index"]))
                    else:
                        cur_form_map = {}
                except Exception as e:
                    logger.warning("[calibrate] Unboosted form computation failed: %s", e)
                    unboosted_map = {}
                    cur_form_map = {}

                # Actual results — build a clean driverId -> position lookup
                evt_actuals = calib_races[
                    (calib_races["season"] == s) & (calib_races["round"] == r)
                ][["driverId", "position"]].drop_duplicates(subset=["driverId"])
                actual_pos_map = dict(zip(evt_actuals["driverId"], evt_actuals["position"]))

                for idx, row in enumerate(X_evt.itertuples()):
                    act_pos = actual_pos_map.get(row.driverId)
                    if act_pos is None or pd.isna(act_pos):
                        continue

                    drv_id = row.driverId
                    calibration_data.append({
                        "driverId": drv_id,
                        "season": s,
                        "round": r,
                        "actual_pos": act_pos,
                        "gbm_raw": pace_hat[idx],
                        "base_form": base_form[idx],
                        "base_team": base_team[idx],
                        "base_dt": base_dt[idx],
                        # Decomposed form: unboosted (all history) + current-season-only
                        "form_unboosted": unboosted_map.get(drv_id, base_form[idx]),
                        "form_cur_season": cur_form_map.get(drv_id, 0.0),
                        "grid": float(X_evt.iloc[idx].get("grid", 15.0)),
                        "elo": elo[idx] if len(elo) > idx else 0,
                        "bt": bt[idx] if len(bt) > idx else 0,
                        "mixed": mixed[idx] if len(mixed) > idx else 0,
                    })

                if (i + 1) % 5 == 0:
                    logger.info("[calibrate] Processed %d/%d events", i + 1, len(events))
                    print(f"    [Calibration] Backtesting event {i + 1}/{len(events)}...")

            if not calibration_data:
                logger.warning("[calibrate] No calibration data generated.")
                print("    [Calibration] ABORTED: No validation queries succeeded.")
                return

            df_calib = pd.DataFrame(calibration_data)

            # 6. Optimisation
            logger.info("[calibrate] Optimising %d parameters on %d samples...", N_PARAMS, len(df_calib))

            # Pre-compute arrays for vectorised objective
            arr_gbm_raw = df_calib["gbm_raw"].values
            arr_base_form = df_calib["base_form"].values
            arr_base_team = df_calib["base_team"].values
            arr_base_dt = df_calib["base_dt"].values
            arr_elo = df_calib["elo"].values
            arr_bt = df_calib["bt"].values
            arr_mixed = df_calib["mixed"].values
            arr_actual_pos = df_calib["actual_pos"].values

            # Decomposed form components for season-weight sensitivity
            arr_form_unboosted = df_calib["form_unboosted"].values
            arr_form_cur_season = df_calib["form_cur_season"].values

            event_keys = df_calib[["season", "round"]].values
            unique_events, event_indices = np.unique(event_keys, axis=0, return_inverse=True)
            n_unique = len(unique_events)

            # Precompute normalised grid per event
            grid_vals = df_calib["grid"].values
            g_z_precomputed = np.zeros_like(grid_vals, dtype=float)
            event_masks = [np.where(event_indices == ei)[0] for ei in range(n_unique)]

            for mask in event_masks:
                ev_grid = grid_vals[mask]
                mu_g = np.nanmean(ev_grid)
                sd_g = np.nanstd(ev_grid)
                g_z_precomputed[mask] = (ev_grid - mu_g) / (sd_g + 1e-6)

            def objective(weights):
                """Combined objective: -mean(per-event Spearman) + podium penalty + regularisation.

                The baseline form component is now recomputed as a function of
                current_season_weight (weights[9]) so the optimizer can learn
                how much to emphasise recent season results vs historical form.
                """
                # Unpack
                wb_gbm = max(0, weights[0])
                wb_form = max(0, weights[1])
                wb_tm = max(0, weights[2])
                wb_dt = max(0, weights[3])
                wb_grid = np.clip(weights[4], 0.0, 1.0)

                we_pace = max(0, weights[5])
                we_elo = max(0, weights[6])
                we_bt = max(0, weights[7])
                we_mixed = max(0, weights[8])

                # Current season weight from optimiser (param index 9)
                w_season = max(1.0, weights[9])

                # Recompute effective form as: unboosted_base + (boost - 1) * cur_season_component
                # When w_season=1.0 this equals unboosted_base (no boost).
                # When w_season=8.0 this adds 7x the current-season-only form.
                # This gives the optimizer a direct gradient on weights[9].
                effective_form = arr_form_unboosted + (w_season - 1.0) * arr_form_cur_season

                # Stage 1: Form-based pace
                raw_pace = (wb_gbm * arr_gbm_raw
                            + wb_form * effective_form
                            + wb_tm * arr_base_team
                            + wb_dt * arr_base_dt)

                grid_imp = np.clip(wb_grid, 0.0, 1.0)

                # Vectorised per-event z-score
                valid_mask = ~np.isnan(raw_pace)
                valid_indices = event_indices[valid_mask]
                valid_pace = raw_pace[valid_mask]

                counts = np.bincount(valid_indices, minlength=n_unique)
                sums = np.bincount(valid_indices, weights=valid_pace, minlength=n_unique)
                safe_counts = np.maximum(counts, 1)
                means = sums / safe_counts

                sq_sums = np.bincount(valid_indices, weights=valid_pace ** 2, minlength=n_unique)
                variances = sq_sums / safe_counts - means ** 2
                variances[variances < 0] = 0
                stds = np.sqrt(variances)

                mu_p_all = means[event_indices]
                sd_p_all = stds[event_indices]
                p_z = (raw_pace - mu_p_all) / (sd_p_all + 1e-6)

                # Stage 2: Grid stickiness blend
                pace = (1.0 - grid_imp) * p_z + grid_imp * g_z_precomputed

                # Ensemble combine
                combined = (we_pace * pace
                            + we_elo * arr_elo
                            + we_bt * arr_bt
                            + we_mixed * arr_mixed)

                # Per-event Spearman rank correlation
                spearman_sum = 0.0
                podium_error_sum = 0.0
                n_events_scored = 0

                for mask in event_masks:
                    if len(mask) < 3:
                        continue
                    pred = combined[mask]
                    actual = arr_actual_pos[mask]
                    # Spearman correlation (we want predicted order to match actual order)
                    if np.std(pred) < 1e-9:
                        continue
                    corr, _ = spearmanr(pred, actual)
                    if not np.isfinite(corr):
                        continue

                    spearman_sum += corr
                    n_events_scored += 1

                    # Podium accuracy: for the 3 drivers predicted fastest,
                    # how far from actual top 3 are they?
                    pred_order = np.argsort(pred)
                    pred_top3_actual = actual[pred_order[:3]]
                    podium_error_sum += np.mean(np.abs(pred_top3_actual - np.arange(1, 4)))

                if n_events_scored == 0:
                    return 1e6

                mean_spearman = spearman_sum / n_events_scored
                mean_podium_err = podium_error_sum / n_events_scored

                # Objective: maximise Spearman (minimise negative), penalise podium error
                # Lambda = 0.02 balances the ~[0,1] Spearman range with ~[0,10] podium error
                loss = -mean_spearman + 0.02 * mean_podium_err

                # Regularisation: The bounds (PARAM_BOUNDS) are the primary
                # guardrails.  We only add a very light L2 penalty to discourage
                # extreme corner solutions when the objective surface is flat,
                # NOT to anchor towards hardcoded "defaults".  The optimizer
                # should learn the right values from data.
                defaults_arr = np.array(PARAM_DEFAULTS, dtype=float)
                scales = np.maximum(np.abs(defaults_arr), 1.0)
                diffs = ((weights - defaults_arr) / scales) ** 2

                # Params 0-9: learnable from the objective — near-zero regularisation
                # Params 10-21: pre-computed (no gradient signal) — light anchor only
                reg_weights = np.ones(N_PARAMS, dtype=float) * 1e-5
                reg_weights[10:] = 0.005
                reg = float(np.sum(reg_weights * diffs))

                return loss + reg

            # Initial guess: always start from PARAM_DEFAULTS so the optimizer
            # learns entirely from the data each time, rather than drifting from
            # a potentially stale cached solution.  The bounds enforce safety.
            init_w = list(PARAM_DEFAULTS)

            res = minimize(objective, init_w, bounds=PARAM_BOUNDS, method="L-BFGS-B",
                           options={"maxiter": 300, "ftol": 1e-9})

            w_opt = res.x
            logger.info("[calibrate] Optimisation result: fun=%.6f, success=%s", res.fun, res.success)

            # Convert to nested structure
            self.current_weights = _unpack_weights(w_opt)
            self.current_weights["objective_score"] = float(res.fun)

            self.save_weights()
            duration = time.time() - t0
            logger.info("[calibrate] Calibration complete. Time: %.1fs", duration)
            print(f"    [Calibration] Complete! Optimised {N_PARAMS} parameters. ({duration:.1f}s)")

        except Exception as e:
            logger.error("[calibrate] Calibration failed: %s", e)
            print(f"    [Calibration] FAILED: {e}")
            import traceback
            traceback.print_exc()
            logger.debug(traceback.format_exc())

    def get_ensemble_config(self) -> EnsembleConfig:
        ens = self.current_weights.get("ensemble", {})
        return EnsembleConfig(
            w_gbm=ens.get("w_gbm", 0.25),
            w_elo=ens.get("w_elo", 0.25),
            w_bt=ens.get("w_bt", 0.25),
            w_mixed=ens.get("w_mixed", 0.25),
        )
