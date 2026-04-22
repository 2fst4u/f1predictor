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
from typing import Dict, Any, Optional, TYPE_CHECKING
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
    # -- ensemble race weights (5-8) --
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
    # -- ensemble qualifying weights (22-25) --
    # Separate weight set for qualifying/sprint_qualifying sessions so that
    # race-only statistical models (Elo, BT, Mixed) can be down-weighted
    # without degrading race predictions.
    "ens_pace_quali",           # 22
    "ens_elo_quali",            # 23
    "ens_bt_quali",             # 24
    "ens_mixed_quali",          # 25
]

N_PARAMS = len(PARAM_NAMES)

# Bounds for L-BFGS-B  (lower, upper) per parameter
PARAM_BOUNDS = [
    (0.05, 1.0),    # 0  gbm_weight
    (0.05, 1.0),    # 1  baseline_weight
    (0.2, 2.0),     # 2  baseline_team_factor
    # TODO: baseline_driver_team_factor is deprecated and fixed at (0.0, 0.0) but still
    # occupies a slot in the optimisation vector, adding minor overhead.  Removing it
    # would require updating _pack/_unpack helpers and the parameter layout.  Requires
    # further investigation and confirmation.
    (0.0, 0.0),     # 3  baseline_driver_team_factor (DEPRECATED — kept for vector compat)
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
    # TODO: dnf_driver_weight and dnf_team_weight are independently bounded [0,1]
    # but not constrained to sum to 1.0 (or any fixed value).  If calibration sets
    # both to 1.0, effective DNF probability is doubled; if both are near 0, DNF
    # becomes negligible.  Adding a sum constraint or normalising these in _unpack_weights
    # could prevent degenerate solutions.  Requires further investigation and confirmation.
    (0.0, 1.0),     # 15 dnf_driver_weight
    (0.0, 1.0),     # 16 dnf_team_weight
    (0.01, 0.5),    # 17 noise_factor
    (0.01, 0.3),    # 18 min_noise
    (30.0, 500.0),  # 19 half_life_base
    (60.0, 800.0),  # 20 half_life_team
    (5.0, 60.0),    # 21 elo_k
    (0.0, 1.0),     # 22 ens_pace_quali
    (0.0, 0.5),     # 23 ens_elo_quali   (capped lower — race-only model)
    (0.0, 0.5),     # 24 ens_bt_quali    (capped lower — race-only model)
    (0.0, 0.5),     # 25 ens_mixed_quali (capped lower — race-only model)
]

# Default / initial guess (matches config.yaml defaults)
PARAM_DEFAULTS = [
    0.75,   # 0  gbm_weight
    0.25,   # 1  baseline_weight
    0.3,    # 2  baseline_team_factor
    0.0,    # 3  baseline_driver_team_factor (DEPRECATED)
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
    0.7,    # 22 ens_pace_quali  — GBM carries most weight for qualifying
    0.1,    # 23 ens_elo_quali
    0.1,    # 24 ens_bt_quali
    0.1,    # 25 ens_mixed_quali
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
        + "v2_form_index"
    )
    return hashlib.sha256(sig.encode()).hexdigest()[:12]


CALIBRATION_VERSION = _calibration_version()


def _unpack_weights(w) -> Dict[str, Any]:
    """Convert flat parameter vector to nested weights dict."""
    import numpy as np
    w = np.asarray(w, dtype=float)

    # Normalise race ensemble weights to sum to 1
    ens_raw = w[5:9].copy()
    ens_sum = ens_raw.sum()
    if ens_sum > 0:
        ens_raw /= ens_sum
    else:
        ens_raw[:] = 0.25

    # Normalise qualifying ensemble weights to sum to 1 (params 22-25)
    # Fall back to defaults if the vector is shorter (old weight files)
    if len(w) > 25:
        ens_q_raw = w[22:26].copy()
    else:
        ens_q_raw = np.array([PARAM_DEFAULTS[22], PARAM_DEFAULTS[23],
                               PARAM_DEFAULTS[24], PARAM_DEFAULTS[25]])
    ens_q_sum = ens_q_raw.sum()
    if ens_q_sum > 0:
        ens_q_raw /= ens_q_sum
    else:
        ens_q_raw[:] = np.array([0.7, 0.1, 0.1, 0.1])

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
            # Qualifying-specific weights
            "w_gbm_quali": float(ens_q_raw[0]),
            "w_elo_quali": float(ens_q_raw[1]),
            "w_bt_quali": float(ens_q_raw[2]),
            "w_mixed_quali": float(ens_q_raw[3]),
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
        e.get("w_gbm_quali", PARAM_DEFAULTS[22]),
        e.get("w_elo_quali", PARAM_DEFAULTS[23]),
        e.get("w_bt_quali", PARAM_DEFAULTS[24]),
        e.get("w_mixed_quali", PARAM_DEFAULTS[25]),
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
        """Save current weights to disk.

        NOTE: The output file (calibration_weights.json) is listed in .gitignore
        and should NOT be committed to the repository.  It is generated at runtime
        and contains environment-specific calibration state (last_race_id, timestamp).
        """
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

            gbm_pipe, _, gbm_features, _ = train_pace_model(X_train, "race", self.cfg)

            # Also train a qualifying-specific GBM for qualifying calibration data
            gbm_pipe_quali = None
            gbm_features_quali = None
            try:
                gbm_pipe_quali, _, gbm_features_quali, _ = train_pace_model(
                    X_train, "qualifying", self.cfg
                )
            except Exception as e:
                logger.warning("[calibrate] Qualifying GBM training failed: %s", e)

            # 5. Collect calibration data from each event.
            #
            # We gather BOTH race and qualifying data so the objective function
            # can simultaneously optimise race weights (params 5-8) and
            # qualifying weights (params 22-25).
            #
            # To allow the optimizer to learn current_season_weight, we store
            # BOTH the boosted form indices AND separate unboosted/current-season
            # components so the objective can recompute form at different boost levels.
            calibration_data = []       # race rows
            quali_calibration_data = [] # qualifying rows

            # Collect qualifying actuals from history (qpos column)
            calib_quali = all_hist[
                (all_hist["session"] == "qualifying")
                & (all_hist["date"] >= start_date)
                & (all_hist["qpos"].notna())
            ].sort_values("date")

            quali_events = calib_quali[["season", "round", "date"]].drop_duplicates().sort_values("date")

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

                # Fit ensemble models once per event; use race track for race data,
                # qualifying track for qualifying data (handled at predict-time below)
                elo_model_fit = EloModel().fit(hist_subset)
                bt_model_fit = BradleyTerryModel().fit(hist_subset)
                mixed_model_fit = MixedEffectsLikeModel().fit(hist_subset)

                elo = elo_model_fit.predict(X_evt, session_type="race")
                bt = bt_model_fit.predict(X_evt, session_type="race")
                mixed = mixed_model_fit.predict(X_evt, session_type="race")

                # Baseline components (boosted form as computed by features.py)
                base_form = -X_evt["form_index"].fillna(0).astype(float).values
                base_team = -X_evt.get("team_form_index", pd.Series(0, index=X_evt.index)).fillna(0).astype(float).values

                # Compute UNBOOSTED form indices for the same drivers so the
                # objective function can recompute form = unboosted + boost * cur_season_only.
                # This gives the optimizer a direct gradient on current_season_weight.
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

                # Actual race results — build a clean driverId -> position lookup
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
                        # Decomposed form: unboosted (all history) + current-season-only
                        "form_unboosted": unboosted_map.get(drv_id, base_form[idx]),
                        "form_cur_season": cur_form_map.get(drv_id, 0.0),
                        "grid": float(X_evt.iloc[idx].get("grid", 15.0)),
                        "elo": elo[idx] if len(elo) > idx else 0,
                        "bt": bt[idx] if len(bt) > idx else 0,
                        "mixed": mixed[idx] if len(mixed) > idx else 0,
                    })

                if (i + 1) % 5 == 0:
                    logger.info("[calibrate] Processed %d/%d race events", i + 1, len(events))
                    print(f"    [Calibration] Backtesting event {i + 1}/{len(events)}...")

            # Collect qualifying calibration data
            logger.info("[calibrate] Generating scores for %d qualifying events...", len(quali_events))
            for i, evt in enumerate(quali_events.itertuples()):
                s, r, d = int(evt.season), int(evt.round), pd.Timestamp(evt.date)

                try:
                    X_evt_q, _, _ = build_session_features(jc, om, s, r, "qualifying", d, self.cfg)
                except Exception as e:
                    logger.debug("[calibrate] Qualifying features failed for %d R%d: %s", s, r, e)
                    continue
                if X_evt_q is None or X_evt_q.empty:
                    continue

                # GBM qualifying score
                try:
                    if gbm_pipe_quali is not None and gbm_features_quali:
                        X_pred_q = X_evt_q.reindex(columns=gbm_features_quali)
                        pace_hat_q = gbm_pipe_quali.predict(X_pred_q)
                    else:
                        X_pred_q = X_evt_q.reindex(columns=gbm_features) if gbm_features else X_evt_q
                        pace_hat_q = gbm_pipe.predict(X_pred_q)
                    pace_hat_q = (pace_hat_q - pace_hat_q.mean()) / (pace_hat_q.std() + 1e-6)
                except Exception as e:
                    logger.debug("[calibrate] Qualifying GBM predict failed for %d R%d: %s", s, r, e)
                    continue

                hist_subset_q = all_hist[all_hist["date"] < d]

                elo_model_q = EloModel().fit(hist_subset_q)
                bt_model_q = BradleyTerryModel().fit(hist_subset_q)
                mixed_model_q = MixedEffectsLikeModel().fit(hist_subset_q)

                elo_q = elo_model_q.predict(X_evt_q, session_type="qualifying")
                bt_q = bt_model_q.predict(X_evt_q, session_type="qualifying")
                mixed_q = mixed_model_q.predict(X_evt_q, session_type="qualifying")

                # Actual qualifying positions
                evt_actuals_q = calib_quali[
                    (calib_quali["season"] == s) & (calib_quali["round"] == r)
                ][["driverId", "qpos"]].drop_duplicates(subset=["driverId"])
                actual_qpos_map = dict(zip(evt_actuals_q["driverId"], evt_actuals_q["qpos"]))

                for idx, row in enumerate(X_evt_q.itertuples()):
                    act_qpos = actual_qpos_map.get(row.driverId)
                    if act_qpos is None or pd.isna(act_qpos):
                        continue
                    quali_calibration_data.append({
                        "driverId": row.driverId,
                        "season": s,
                        "round": r,
                        "actual_pos": act_qpos,
                        "gbm_raw": pace_hat_q[idx],
                        "elo": elo_q[idx] if len(elo_q) > idx else 0,
                        "bt": bt_q[idx] if len(bt_q) > idx else 0,
                        "mixed": mixed_q[idx] if len(mixed_q) > idx else 0,
                    })

            if not calibration_data:
                logger.warning("[calibrate] No race calibration data generated.")
                print("    [Calibration] ABORTED: No validation queries succeeded.")
                return

            df_calib = pd.DataFrame(calibration_data)
            df_calib_q = pd.DataFrame(quali_calibration_data) if quali_calibration_data else pd.DataFrame()
            logger.info(
                "[calibrate] Race samples: %d; Qualifying samples: %d",
                len(df_calib), len(df_calib_q),
            )

            # 6. Optimisation
            logger.info("[calibrate] Optimising %d parameters on %d race + %d qualifying samples...",
                        N_PARAMS, len(df_calib), len(df_calib_q))

            # Pre-compute arrays for vectorised race objective
            arr_gbm_raw = df_calib["gbm_raw"].values
            _ = df_calib["base_form"].values
            arr_base_team = df_calib["base_team"].values
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

            # Pre-compute arrays for qualifying objective (may be empty)
            has_quali_data = not df_calib_q.empty
            if has_quali_data:
                arr_q_gbm = df_calib_q["gbm_raw"].values
                arr_q_elo = df_calib_q["elo"].values
                arr_q_bt = df_calib_q["bt"].values
                arr_q_mixed = df_calib_q["mixed"].values
                arr_q_actual = df_calib_q["actual_pos"].values
                q_event_keys = df_calib_q[["season", "round"]].values
                _, q_event_indices = np.unique(q_event_keys, axis=0, return_inverse=True)
                n_q_unique = len(np.unique(q_event_keys, axis=0))
                q_event_masks = [
                    np.where(q_event_indices == ei)[0] for ei in range(n_q_unique)
                ]

            def objective(weights):
                """Combined objective over race AND qualifying sessions.

                Race term: -Spearman(predicted_pace, actual_race_pos) using race weights (5-8)
                Qualifying term: -Spearman(predicted_pace, actual_qpos) using quali weights (22-25)

                The two terms are averaged with equal weight so the optimizer
                simultaneously learns good race predictions and good qualifying
                predictions, including the correct trade-off between the GBM
                and the statistical models for each session type.

                TODO: This objective uses a simplified surrogate of the production
                pipeline (static GBM predictions, simplified grid stickiness blend
                instead of the anchor-delta model in models.py).  Parameters like
                half_life_base, current_season_weight, and grid_factor are optimised
                against a different pipeline than the one that uses them at inference
                time.  Aligning this objective with the production logic could improve
                calibration fidelity.  Requires further investigation and confirmation.
                """
                # Unpack race/blending params
                wb_gbm = max(0, weights[0])
                wb_form = max(0, weights[1])
                wb_tm = max(0, weights[2])
                wb_grid = np.clip(weights[4], 0.0, 1.0)

                we_pace = max(0, weights[5])
                we_elo = max(0, weights[6])
                we_bt = max(0, weights[7])
                we_mixed = max(0, weights[8])

                # Qualifying ensemble weights (params 22-25)
                we_q_pace = max(0, weights[22]) if len(weights) > 22 else PARAM_DEFAULTS[22]
                we_q_elo = max(0, weights[23]) if len(weights) > 23 else PARAM_DEFAULTS[23]
                we_q_bt = max(0, weights[24]) if len(weights) > 24 else PARAM_DEFAULTS[24]
                we_q_mixed = max(0, weights[25]) if len(weights) > 25 else PARAM_DEFAULTS[25]

                # Current season weight from optimiser (param index 9)
                w_season = max(1.0, weights[9])

                # Recompute effective form as: unboosted_base + (boost - 1) * cur_season_component
                effective_form = arr_form_unboosted + (w_season - 1.0) * arr_form_cur_season

                # ---- Race objective ----
                raw_pace = (wb_gbm * arr_gbm_raw
                            + wb_form * effective_form
                            + wb_tm * arr_base_team)

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

                # Grid stickiness blend
                pace = (1.0 - grid_imp) * p_z + grid_imp * g_z_precomputed

                # Normalise race ensemble weights
                race_wsum = we_pace + we_elo + we_bt + we_mixed
                if race_wsum < 1e-9:
                    race_wsum = 1.0
                combined_race = (
                    (we_pace / race_wsum) * pace
                    + (we_elo / race_wsum) * arr_elo
                    + (we_bt / race_wsum) * arr_bt
                    + (we_mixed / race_wsum) * arr_mixed
                )

                race_spearman_sum = 0.0
                race_podium_err_sum = 0.0
                n_race_scored = 0

                for mask in event_masks:
                    if len(mask) < 3:
                        continue
                    pred = combined_race[mask]
                    actual = arr_actual_pos[mask]
                    if np.std(pred) < 1e-9:
                        continue
                    corr, _ = spearmanr(pred, actual)
                    if not np.isfinite(corr):
                        continue
                    race_spearman_sum += corr
                    n_race_scored += 1
                    pred_order = np.argsort(pred)
                    pred_top3_actual = actual[pred_order[:3]]
                    race_podium_err_sum += np.mean(np.abs(pred_top3_actual - np.arange(1, 4)))

                if n_race_scored == 0:
                    return 1e6

                mean_race_spearman = race_spearman_sum / n_race_scored
                mean_race_podium_err = race_podium_err_sum / n_race_scored
                race_loss = -mean_race_spearman + 0.02 * mean_race_podium_err

                # ---- Qualifying objective ----
                quali_loss = 0.0
                if has_quali_data:
                    # Normalise qualifying ensemble weights
                    q_wsum = we_q_pace + we_q_elo + we_q_bt + we_q_mixed
                    if q_wsum < 1e-9:
                        q_wsum = 1.0
                    combined_quali = (
                        (we_q_pace / q_wsum) * arr_q_gbm
                        + (we_q_elo / q_wsum) * arr_q_elo
                        + (we_q_bt / q_wsum) * arr_q_bt
                        + (we_q_mixed / q_wsum) * arr_q_mixed
                    )

                    q_spearman_sum = 0.0
                    q_podium_err_sum = 0.0
                    n_q_scored = 0

                    for mask in q_event_masks:
                        if len(mask) < 3:
                            continue
                        pred_q = combined_quali[mask]
                        actual_q = arr_q_actual[mask]
                        if np.std(pred_q) < 1e-9:
                            continue
                        corr_q, _ = spearmanr(pred_q, actual_q)
                        if not np.isfinite(corr_q):
                            continue
                        q_spearman_sum += corr_q
                        n_q_scored += 1
                        pred_q_order = np.argsort(pred_q)
                        pred_q_top3_actual = actual_q[pred_q_order[:3]]
                        q_podium_err_sum += np.mean(np.abs(pred_q_top3_actual - np.arange(1, 4)))

                    if n_q_scored > 0:
                        mean_q_spearman = q_spearman_sum / n_q_scored
                        mean_q_podium_err = q_podium_err_sum / n_q_scored
                        quali_loss = -mean_q_spearman + 0.02 * mean_q_podium_err

                # Combined loss: equal weighting of race and qualifying performance
                if has_quali_data and quali_loss != 0.0:
                    total_loss = 0.5 * race_loss + 0.5 * quali_loss
                else:
                    total_loss = race_loss

                # Regularisation: very light L2 to discourage extreme corner solutions.
                # The bounds (PARAM_BOUNDS) are the primary guardrails; regularisation
                # only prevents degenerate solutions at the boundary edges.
                defaults_arr = np.array(PARAM_DEFAULTS, dtype=float)
                scales = np.maximum(np.abs(defaults_arr), 1.0)
                diffs = ((weights - defaults_arr) / scales) ** 2

                # Minimal regularisation — just enough to avoid degenerate solutions.
                # Previously heavier anchoring (1e-5 / 0.005) prevented the optimizer
                # from diverging from defaults.  The bounds already constrain the
                # search space, so regularisation should only act as a tiebreaker.
                reg_weights = np.ones(N_PARAMS, dtype=float) * 1e-7
                reg_weights[10:22] = 1e-5
                reg = float(np.sum(reg_weights * diffs))

                return total_loss + reg

            # Initial guess: always start from PARAM_DEFAULTS so the optimizer
            # learns entirely from the data each time, rather than drifting from
            # a potentially stale cached solution.  The bounds enforce safety.
            init_w = list(PARAM_DEFAULTS)

            # Multi-start optimisation: run L-BFGS-B from the default starting
            # point AND from several perturbed initialisations to avoid getting
            # trapped in the basin around the defaults.  The best result wins.
            best_res = None

            def _run_lbfgsb(x0, label):
                return minimize(
                    objective, x0, bounds=PARAM_BOUNDS, method="L-BFGS-B",
                    options={"maxiter": 800, "ftol": 1e-12, "gtol": 1e-8},
                )

            # Start 0: from defaults
            res0 = _run_lbfgsb(init_w, "defaults")
            best_res = res0
            logger.info("[calibrate] Start 0 (defaults): fun=%.6f success=%s", res0.fun, res0.success)

            # Starts 1-4: perturbed initialisations within bounds
            rng = np.random.RandomState(42)
            for start_idx in range(1, 5):
                perturbed = np.array(init_w, dtype=float)
                noise = rng.uniform(-0.15, 0.15, size=N_PARAMS)
                for pi in range(N_PARAMS):
                    lo, hi = PARAM_BOUNDS[pi]
                    span = hi - lo
                    if span > 0:
                        perturbed[pi] = np.clip(
                            perturbed[pi] + noise[pi] * span,
                            lo, hi,
                        )
                res_i = _run_lbfgsb(perturbed.tolist(), f"perturbed-{start_idx}")
                logger.info(
                    "[calibrate] Start %d (perturbed): fun=%.6f success=%s",
                    start_idx, res_i.fun, res_i.success,
                )
                if res_i.fun < best_res.fun:
                    best_res = res_i

            w_opt = best_res.x
            logger.info(
                "[calibrate] Best optimisation result: fun=%.6f, success=%s (from %d starts)",
                best_res.fun, best_res.success, 5,
            )

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
            w_gbm=ens.get("w_gbm", 0.4),
            w_elo=ens.get("w_elo", 0.2),
            w_bt=ens.get("w_bt", 0.2),
            w_mixed=ens.get("w_mixed", 0.2),
            w_gbm_quali=ens.get("w_gbm_quali", PARAM_DEFAULTS[22]),
            w_elo_quali=ens.get("w_elo_quali", PARAM_DEFAULTS[23]),
            w_bt_quali=ens.get("w_bt_quali", PARAM_DEFAULTS[24]),
            w_mixed_quali=ens.get("w_mixed_quali", PARAM_DEFAULTS[25]),
        )
