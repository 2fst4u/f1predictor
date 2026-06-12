"""Dynamic calibration of model parameters based on recent history.

This module provides the CalibrationManager to optimize the tunable weights
by minimizing prediction error over a sliding lookback window.  The objective
combines per-event rank terms (Spearman + podium error) with probability
calibration terms (analytic pairwise/winner Brier mirroring the Monte Carlo
noise model, and a DNF Brier using the exact estimate_dnf_probabilities
formula), so ranking AND probability outputs are both fitted to outcomes.

Every parameter in the optimisation vector receives real gradient signal from
the objective.  Parameters that cannot be exercised without re-fitting all
precomputed components per candidate (recency half-lives, Elo K, the sprint
season weight) are NOT optimised: they are emitted as fixed defaults (see
FIXED_DEFAULTS) so the weights-file schema stays complete and production
behaviour stays explicit.

Calibrated parameter groups:
    - blending: GBM/baseline weights, team factor, grid stickiness,
                season weights, qualifying factor, analytical win weight
    - ensemble: w_gbm/w_elo/w_bt/w_mixed (race + qualifying sets)
    - dnf:      alpha, beta, driver_weight, team_weight
    - simulation: noise_factor, min_noise
Fixed (not optimised, emitted as defaults):
    - recency:  half_life_base, half_life_team
    - elo:      k (Elo update strength)
    - blending.current_season_sprint_weight
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

# Every parameter in this vector receives a real gradient from the objective
# function (rank terms, probability terms, or the DNF term).  Parameters that
# CANNOT be exercised cheaply by the objective — recency half-lives, Elo K and
# the sprint season weight — are intentionally NOT in the vector: they are
# emitted as fixed defaults (see FIXED_DEFAULTS) instead of pretending to be
# calibrated while only ever being moved by the regulariser.
PARAM_NAMES = [
    # -- blending (0-3) --
    "gbm_weight",               # 0
    "baseline_weight",          # 1
    "baseline_team_factor",     # 2
    "grid_factor",              # 3
    # -- ensemble race weights (4-7) --
    "ens_pace",                 # 4
    "ens_elo",                  # 5
    "ens_bt",                   # 6
    "ens_mixed",                # 7
    # -- season / quali (8-11) --
    "current_season_weight",    # 8
    "current_season_qualifying_weight",  # 9
    "current_quali_factor",     # 10
    "analytical_win_weight",    # 11
    # -- DNF (12-15) --
    "dnf_alpha",                # 12
    "dnf_beta",                 # 13
    "dnf_driver_weight",        # 14
    "dnf_team_weight",          # 15
    # -- simulation (16-17) --
    "noise_factor",             # 16
    "min_noise",                # 17
    # -- ensemble qualifying weights (18-21) --
    # Separate weight set for qualifying/sprint_qualifying sessions so that
    # race-only statistical models (Elo, BT, Mixed) can be down-weighted
    # without degrading race predictions.
    "ens_pace_quali",           # 18
    "ens_elo_quali",            # 19
    "ens_bt_quali",             # 20
    "ens_mixed_quali",          # 21
]

N_PARAMS = len(PARAM_NAMES)

# Bounds for L-BFGS-B  (lower, upper) per parameter
PARAM_BOUNDS = [
    (0.05, 1.0),    # 0  gbm_weight
    (0.05, 1.0),    # 1  baseline_weight
    (0.2, 2.0),     # 2  baseline_team_factor
    # grid_factor bounds match the production clamp in models.py
    # (dynamic_w_grid is clipped to [0.4, 0.95]); allowing lower values here
    # would calibrate a number production silently overrides.
    (0.4, 0.95),    # 3  grid_factor
    (0.0, 1.0),     # 4  ens_pace
    (0.0, 1.0),     # 5  ens_elo
    (0.0, 1.0),     # 6  ens_bt
    (0.0, 1.0),     # 7  ens_mixed
    (1.0, 50.0),    # 8  current_season_weight
    (1.0, 50.0),    # 9  current_season_qualifying_weight
    (0.0, 1.0),     # 10 current_quali_factor
    (0.0, 1.0),     # 11 analytical_win_weight
    (0.1, 10.0),    # 12 dnf_alpha
    (0.1, 30.0),    # 13 dnf_beta
    # driver/team DNF weights are normalised to a convex blend downstream
    # (estimate_dnf_probabilities divides by their sum), so only their ratio
    # matters; bounds [0,1] are safe.
    (0.0, 1.0),     # 14 dnf_driver_weight
    (0.0, 1.0),     # 15 dnf_team_weight
    (0.01, 0.5),    # 16 noise_factor
    (0.01, 0.3),    # 17 min_noise
    (0.0, 1.0),     # 18 ens_pace_quali
    (0.0, 0.5),     # 19 ens_elo_quali   (capped lower — race-only model)
    (0.0, 0.5),     # 20 ens_bt_quali    (capped lower — race-only model)
    (0.0, 0.5),     # 21 ens_mixed_quali (capped lower — race-only model)
]

# Default / initial guess (matches config.yaml defaults)
PARAM_DEFAULTS = [
    0.75,   # 0  gbm_weight
    0.25,   # 1  baseline_weight
    0.3,    # 2  baseline_team_factor
    0.8,    # 3  grid_factor
    0.4,    # 4  ens_pace
    0.2,    # 5  ens_elo
    0.2,    # 6  ens_bt
    0.2,    # 7  ens_mixed
    8.0,    # 8  current_season_weight
    8.0,    # 9  current_season_qualifying_weight
    0.5,    # 10 current_quali_factor
    0.5,    # 11 analytical_win_weight
    2.0,    # 12 dnf_alpha
    8.0,    # 13 dnf_beta
    0.6,    # 14 dnf_driver_weight
    0.4,    # 15 dnf_team_weight
    0.15,   # 16 noise_factor
    0.05,   # 17 min_noise
    0.7,    # 18 ens_pace_quali  — GBM carries most weight for qualifying
    0.1,    # 19 ens_elo_quali
    0.1,    # 20 ens_bt_quali
    0.1,    # 21 ens_mixed_quali
]

# Parameters used by production but NOT optimised here, because the objective
# cannot give them a gradient without re-fitting all components per candidate:
#   - recency half-lives feed every precomputed component (form, Elo, BT, Mixed)
#   - elo_k changes the Elo fit itself
#   - the sprint season weight has too few sprint events to fit reliably
# They are emitted verbatim so the weights-file schema stays complete.
FIXED_DEFAULTS = {
    "recency": {"half_life_base": 120.0, "half_life_team": 240.0},
    "elo": {"k": 20.0},
    "current_season_sprint_weight": 8.0,
}


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
        + "v3_outcome_targets_prob_calibration"
    )
    return hashlib.sha256(sig.encode()).hexdigest()[:12]


CALIBRATION_VERSION = _calibration_version()


def _unpack_weights(w) -> Dict[str, Any]:
    """Convert flat parameter vector to nested weights dict.

    The recency, elo, and sprint-weight entries are emitted from
    FIXED_DEFAULTS — they are structural constants, not optimised values
    (see the note above PARAM_NAMES).
    """
    import numpy as np
    w = np.asarray(w, dtype=float)

    # Normalise race ensemble weights to sum to 1
    ens_raw = w[4:8].copy()
    ens_sum = ens_raw.sum()
    if ens_sum > 0:
        ens_raw /= ens_sum
    else:
        ens_raw[:] = 0.25

    # Normalise qualifying ensemble weights to sum to 1 (params 18-21)
    if len(w) > 21:
        ens_q_raw = w[18:22].copy()
    else:
        ens_q_raw = np.array([PARAM_DEFAULTS[18], PARAM_DEFAULTS[19],
                              PARAM_DEFAULTS[20], PARAM_DEFAULTS[21]])
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
            "grid_factor": float(np.clip(w[3], 0.0, 1.0)),
            "current_season_weight": float(w[8]),
            "current_season_qualifying_weight": float(w[9]),
            "current_quali_factor": float(np.clip(w[10], 0.0, 1.0)),
            "analytical_win_weight": float(np.clip(w[11], 0.0, 1.0)),
            "current_season_sprint_weight": float(FIXED_DEFAULTS["current_season_sprint_weight"]),
        },
        "dnf": {
            "alpha": float(max(0.1, w[12])),
            "beta": float(max(0.1, w[13])),
            "driver_weight": float(np.clip(w[14], 0.0, 1.0)),
            "team_weight": float(np.clip(w[15], 0.0, 1.0)),
        },
        "simulation": {
            "noise_factor": float(max(0.01, w[16])),
            "min_noise": float(max(0.01, w[17])),
        },
        "recency": dict(FIXED_DEFAULTS["recency"]),
        "elo": dict(FIXED_DEFAULTS["elo"]),
    }


def _pack_weights(d: Dict[str, Any]) -> list:
    """Convert nested weights dict back to flat parameter vector."""
    b = d.get("blending", {})
    e = d.get("ensemble", {})
    dn = d.get("dnf", {})
    sim = d.get("simulation", {})

    return [
        b.get("gbm_weight", PARAM_DEFAULTS[0]),
        b.get("baseline_weight", PARAM_DEFAULTS[1]),
        b.get("baseline_team_factor", PARAM_DEFAULTS[2]),
        b.get("grid_factor", PARAM_DEFAULTS[3]),
        e.get("w_gbm", PARAM_DEFAULTS[4]),
        e.get("w_elo", PARAM_DEFAULTS[5]),
        e.get("w_bt", PARAM_DEFAULTS[6]),
        e.get("w_mixed", PARAM_DEFAULTS[7]),
        b.get("current_season_weight", PARAM_DEFAULTS[8]),
        b.get("current_season_qualifying_weight", PARAM_DEFAULTS[9]),
        b.get("current_quali_factor", PARAM_DEFAULTS[10]),
        b.get("analytical_win_weight", PARAM_DEFAULTS[11]),
        dn.get("alpha", PARAM_DEFAULTS[12]),
        dn.get("beta", PARAM_DEFAULTS[13]),
        dn.get("driver_weight", PARAM_DEFAULTS[14]),
        dn.get("team_weight", PARAM_DEFAULTS[15]),
        sim.get("noise_factor", PARAM_DEFAULTS[16]),
        sim.get("min_noise", PARAM_DEFAULTS[17]),
        e.get("w_gbm_quali", PARAM_DEFAULTS[18]),
        e.get("w_elo_quali", PARAM_DEFAULTS[19]),
        e.get("w_bt_quali", PARAM_DEFAULTS[20]),
        e.get("w_mixed_quali", PARAM_DEFAULTS[21]),
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

            # 3. Train GBM Baseline on pre-window data
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

            # Train the calibration GBM the same way production does: on an
            # out-of-sample historical matrix with ACTUAL outcomes (y_pace)
            # as the target.  Falls back to the in-sample fit only when the
            # historical matrix cannot be built.
            hist_X_cal = None
            try:
                from .models import build_hist_training_X
                hist_X_cal = build_hist_training_X(
                    train_hist, X_train, train_cutoff,
                    half_life_days=self.cfg.modelling.recency_half_life_days.base,
                )
            except Exception as e:
                logger.warning("[calibrate] Historical training matrix failed: %s", e)

            gbm_pipe, _, gbm_features, _ = train_pace_model(
                X_train, "race", self.cfg, hist_X=hist_X_cal,
            )

            # Also train a qualifying-specific GBM for qualifying calibration data
            gbm_pipe_quali = None
            gbm_features_quali = None
            try:
                gbm_pipe_quali, _, gbm_features_quali, _ = train_pace_model(
                    X_train, "qualifying", self.cfg, hist_X=hist_X_cal,
                )
            except Exception as e:
                logger.warning("[calibrate] Qualifying GBM training failed: %s", e)

            # 4. Collect calibration data from each event.
            #
            # We gather BOTH race and qualifying data so the objective function
            # can simultaneously optimise race weights (params 4-7) and
            # qualifying weights (params 21-24).
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
                # qualifying track for qualifying data (handled at predict-time below).
                # Elo uses the team recency half-life so historical ratings decay
                # toward the mean, matching the inference path in predict.py.
                elo_model_fit = EloModel().fit(
                    hist_subset,
                    half_life_days=self.cfg.modelling.recency_half_life_days.team,
                )
                bt_model_fit = BradleyTerryModel().fit(hist_subset)
                mixed_model_fit = MixedEffectsLikeModel().fit(hist_subset)

                elo = elo_model_fit.predict(X_evt, session_type="race")
                bt = bt_model_fit.predict(X_evt, session_type="race")
                mixed = mixed_model_fit.predict(X_evt, session_type="race")

                # Baseline components (boosted form as computed by features.py)
                base_form = -X_evt["form_index"].fillna(0).astype(float).values
                base_team = -X_evt.get("team_form_index", pd.Series(0, index=X_evt.index)).fillna(0).astype(float).values

                # Weighted-sum form components so the objective can recompute the
                # EXACT production form formula for any current_season_weight b:
                #     form(b) = (s_pre + b*s_cur) / (w_pre + b*w_cur)
                # (production applies the boost as a multiplicative weight inside
                # a weighted average — a ratio, not an additive term).
                try:
                    from .features import compute_form_components
                    hl_base = self.cfg.modelling.recency_half_life_days.base
                    comp_df = compute_form_components(
                        hist_subset, ref_date=d, half_life_days=hl_base, current_season=s,
                    )
                    comp_map = {
                        row_c.driverId: (row_c.s_pre, row_c.w_pre, row_c.s_cur, row_c.w_cur)
                        for row_c in comp_df.itertuples()
                    }
                except Exception as e:
                    logger.warning("[calibrate] Form component computation failed: %s", e)
                    comp_map = {}

                # Per-driver / per-team DNF base-rate counts BEFORE this event, so
                # the objective can recompute estimate_dnf_probabilities exactly
                # for any (alpha, beta, driver_weight, team_weight) candidate.
                from .features import compute_dnf_flags
                races_h = hist_subset[hist_subset["session"] == "race"]
                if not races_h.empty:
                    if "is_dnf" in races_h.columns:
                        h_dnf = races_h["is_dnf"].astype(float).values
                    else:
                        h_dnf = compute_dnf_flags(races_h).astype(float).values
                    h_tmp = races_h.assign(_dnf=h_dnf)
                    drv_stats = h_tmp.groupby("driverId")["_dnf"].agg(["sum", "count"])
                    team_stats = (
                        h_tmp.dropna(subset=["constructorId"]).groupby("constructorId")["_dnf"].agg(["sum", "count"])
                        if "constructorId" in h_tmp.columns else None
                    )
                    dnf_gk, dnf_gn = float(h_tmp["_dnf"].sum()), int(len(h_tmp))
                else:
                    drv_stats = team_stats = None
                    dnf_gk, dnf_gn = 0.0, 0

                # Circuit DNF modifier exactly as estimate_dnf_probabilities uses it
                circuit_mod = 1.0
                if "global_circuit_dnf_rate" in X_evt.columns:
                    c_mean = X_evt["global_circuit_dnf_rate"].mean()
                    if pd.notna(c_mean):
                        circuit_mod = float(c_mean) / 0.08

                # Actual race results — position AND DNF flag per driver
                evt_rows = all_hist[
                    (all_hist["season"] == s) & (all_hist["round"] == r)
                    & (all_hist["session"] == "race")
                ].drop_duplicates(subset=["driverId"])
                if not evt_rows.empty:
                    if "is_dnf" in evt_rows.columns:
                        evt_dnf_vals = evt_rows["is_dnf"].astype(float).values
                    else:
                        evt_dnf_vals = compute_dnf_flags(evt_rows).astype(float).values
                    actual_dnf_map = dict(zip(evt_rows["driverId"], evt_dnf_vals))
                else:
                    actual_dnf_map = {}
                evt_actuals = calib_races[
                    (calib_races["season"] == s) & (calib_races["round"] == r)
                ][["driverId", "position"]].drop_duplicates(subset=["driverId"])
                actual_pos_map = dict(zip(evt_actuals["driverId"], evt_actuals["position"]))

                for idx, row in enumerate(X_evt.itertuples()):
                    act_pos = actual_pos_map.get(row.driverId)
                    if act_pos is None or pd.isna(act_pos):
                        continue

                    drv_id = row.driverId
                    team_id = getattr(row, "constructorId", None)
                    s_pre, w_pre, s_cur, w_cur = comp_map.get(drv_id, (0.0, 0.0, 0.0, 0.0))
                    d_k, d_n = (
                        (float(drv_stats.loc[drv_id, "sum"]), float(drv_stats.loc[drv_id, "count"]))
                        if drv_stats is not None and drv_id in drv_stats.index else (0.0, 0.0)
                    )
                    t_k, t_n = (
                        (float(team_stats.loc[team_id, "sum"]), float(team_stats.loc[team_id, "count"]))
                        if team_stats is not None and team_id in team_stats.index else (0.0, 0.0)
                    )
                    calibration_data.append({
                        "driverId": drv_id,
                        "season": s,
                        "round": r,
                        "actual_pos": act_pos,
                        "actual_dnf": float(actual_dnf_map.get(drv_id, 0.0)),
                        "gbm_raw": pace_hat[idx],
                        "base_form": base_form[idx],
                        "base_team": base_team[idx],
                        # Exact production form components for season-weight gradient
                        "form_s_pre": float(s_pre), "form_w_pre": float(w_pre),
                        "form_s_cur": float(s_cur), "form_w_cur": float(w_cur),
                        "grid": float(X_evt.iloc[idx].get("grid", np.nan)),
                        # Current-weekend qualifying position (NaN when unavailable);
                        # lets the objective exercise current_quali_factor exactly as
                        # the production pace blend in models.py does.
                        "current_quali_pos": float(X_evt.iloc[idx].get("current_quali_pos", np.nan)),
                        "elo": elo[idx] if len(elo) > idx else 0,
                        "bt": bt[idx] if len(bt) > idx else 0,
                        "mixed": mixed[idx] if len(mixed) > idx else 0,
                        # DNF base-rate sufficient statistics
                        "dnf_drv_k": d_k, "dnf_drv_n": d_n,
                        "dnf_team_k": t_k, "dnf_team_n": t_n,
                        "dnf_global_k": dnf_gk, "dnf_global_n": float(dnf_gn),
                        "dnf_circuit_mod": circuit_mod,
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

                elo_model_q = EloModel().fit(
                    hist_subset_q,
                    half_life_days=self.cfg.modelling.recency_half_life_days.team,
                )
                bt_model_q = BradleyTerryModel().fit(hist_subset_q)
                mixed_model_q = MixedEffectsLikeModel().fit(hist_subset_q)

                elo_q = elo_model_q.predict(X_evt_q, session_type="qualifying")
                bt_q = bt_model_q.predict(X_evt_q, session_type="qualifying")
                mixed_q = mixed_model_q.predict(X_evt_q, session_type="qualifying")

                base_team_q = -X_evt_q.get("team_form_index", pd.Series(0, index=X_evt_q.index)).fillna(0).astype(float).values

                # Qualifying form components (qpos-based) so the objective can
                # exercise current_season_qualifying_weight with the exact
                # production ratio formula.
                try:
                    from .features import compute_form_components
                    hl_base = self.cfg.modelling.recency_half_life_days.base
                    comp_q_df = compute_form_components(
                        hist_subset_q, ref_date=d, half_life_days=hl_base,
                        current_season=s,
                        sessions=("qualifying", "sprint_qualifying"), pos_col="qpos",
                    )
                    comp_q_map = {
                        row_c.driverId: (row_c.s_pre, row_c.w_pre, row_c.s_cur, row_c.w_cur)
                        for row_c in comp_q_df.itertuples()
                    }
                except Exception as e:
                    logger.debug("[calibrate] Quali form components failed: %s", e)
                    comp_q_map = {}

                # Actual qualifying positions
                evt_actuals_q = calib_quali[
                    (calib_quali["season"] == s) & (calib_quali["round"] == r)
                ][["driverId", "qpos"]].drop_duplicates(subset=["driverId"])
                actual_qpos_map = dict(zip(evt_actuals_q["driverId"], evt_actuals_q["qpos"]))

                for idx, row in enumerate(X_evt_q.itertuples()):
                    act_qpos = actual_qpos_map.get(row.driverId)
                    if act_qpos is None or pd.isna(act_qpos):
                        continue
                    s_pre, w_pre, s_cur, w_cur = comp_q_map.get(row.driverId, (0.0, 0.0, 0.0, 0.0))
                    quali_calibration_data.append({
                        "driverId": row.driverId,
                        "season": s,
                        "round": r,
                        "actual_pos": act_qpos,
                        "gbm_raw": pace_hat_q[idx],
                        "base_team": base_team_q[idx],
                        "form_s_pre": float(s_pre), "form_w_pre": float(w_pre),
                        "form_s_cur": float(s_cur), "form_w_cur": float(w_cur),
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

            # 5. Optimisation
            logger.info("[calibrate] Optimising %d parameters on %d race + %d qualifying samples...",
                        N_PARAMS, len(df_calib), len(df_calib_q))

            from scipy.special import ndtr
            from .simulate import NOISE_STD_MULTIPLIER

            # Pre-compute arrays for vectorised race objective
            arr_gbm_raw = df_calib["gbm_raw"].values.astype(float)
            arr_base_team = df_calib["base_team"].values.astype(float)
            arr_elo = df_calib["elo"].values.astype(float)
            arr_bt = df_calib["bt"].values.astype(float)
            arr_mixed = df_calib["mixed"].values.astype(float)
            arr_actual_pos = df_calib["actual_pos"].values.astype(float)
            arr_actual_dnf = df_calib["actual_dnf"].values.astype(float)

            # Form sufficient statistics for the exact production boost formula
            arr_s_pre = df_calib["form_s_pre"].values.astype(float)
            arr_w_pre = df_calib["form_w_pre"].values.astype(float)
            arr_s_cur = df_calib["form_s_cur"].values.astype(float)
            arr_w_cur = df_calib["form_w_cur"].values.astype(float)

            # DNF sufficient statistics
            arr_dnf_drv_k = df_calib["dnf_drv_k"].values.astype(float)
            arr_dnf_drv_n = df_calib["dnf_drv_n"].values.astype(float)
            arr_dnf_team_k = df_calib["dnf_team_k"].values.astype(float)
            arr_dnf_team_n = df_calib["dnf_team_n"].values.astype(float)
            arr_dnf_gk = df_calib["dnf_global_k"].values.astype(float)
            arr_dnf_gn = df_calib["dnf_global_n"].values.astype(float)
            arr_dnf_cmod = df_calib["dnf_circuit_mod"].values.astype(float)
            dnf_clip_min = float(getattr(getattr(self.cfg.modelling, "dnf", None), "clip_min", 0.02) or 0.02) \
                if hasattr(self.cfg, "modelling") else 0.02
            dnf_clip_max = float(getattr(getattr(self.cfg.modelling, "dnf", None), "clip_max", 0.35) or 0.35) \
                if hasattr(self.cfg, "modelling") else 0.35

            event_keys = df_calib[["season", "round"]].values
            unique_events, event_indices = np.unique(event_keys, axis=0, return_inverse=True)
            n_unique = len(unique_events)
            event_masks = [np.where(event_indices == ei)[0] for ei in range(n_unique)]

            def _event_z(vals: np.ndarray, ev_idx: np.ndarray, n_ev: int) -> np.ndarray:
                """Vectorised per-event z-score; NaNs contribute 0 and stay 0."""
                valid = np.isfinite(vals)
                safe_vals = np.where(valid, vals, 0.0)
                counts = np.bincount(ev_idx[valid], minlength=n_ev)
                sums = np.bincount(ev_idx[valid], weights=safe_vals[valid], minlength=n_ev)
                safe_counts = np.maximum(counts, 1)
                means = sums / safe_counts
                sq_sums = np.bincount(ev_idx[valid], weights=safe_vals[valid] ** 2, minlength=n_ev)
                variances = sq_sums / safe_counts - means ** 2
                variances[variances < 0] = 0
                stds = np.sqrt(variances)
                z = (vals - means[ev_idx]) / (stds[ev_idx] + 1e-6)
                return np.where(valid, z, 0.0)

            # Grid filled with the event size for drivers without a grid slot
            # (mirrors models.py grid fallback of len(X)).
            grid_vals = df_calib["grid"].values.astype(float)
            event_sizes = np.bincount(event_indices, minlength=n_unique).astype(float)
            grid_filled = np.where(np.isfinite(grid_vals), grid_vals, event_sizes[event_indices])

            # Precompute normalised current-weekend qualifying position per event so
            # the objective can reproduce the production current_quali blend.  Drivers
            # without a current-quali result keep their raw pace (mask = False).
            if "current_quali_pos" in df_calib.columns:
                quali_pos_vals = df_calib["current_quali_pos"].values.astype(float)
            else:
                quali_pos_vals = np.full(len(arr_gbm_raw), np.nan, dtype=float)
            quali_blend_mask = ~np.isnan(quali_pos_vals)
            q_z_precomputed = np.zeros_like(quali_pos_vals, dtype=float)
            for mask in event_masks:
                vals = quali_pos_vals[mask]
                valid = ~np.isnan(vals)
                if valid.sum() < 2:
                    continue
                mu_q = np.nanmean(vals)
                sd_q = np.nanstd(vals)
                q_z_precomputed[mask] = np.where(
                    valid, (vals - mu_q) / (sd_q + 1e-6), 0.0
                )

            # Pre-compute arrays for qualifying objective (may be empty)
            has_quali_data = not df_calib_q.empty
            if has_quali_data:
                arr_q_gbm = df_calib_q["gbm_raw"].values.astype(float)
                arr_q_team = df_calib_q["base_team"].values.astype(float)
                arr_q_elo = df_calib_q["elo"].values.astype(float)
                arr_q_bt = df_calib_q["bt"].values.astype(float)
                arr_q_mixed = df_calib_q["mixed"].values.astype(float)
                arr_q_actual = df_calib_q["actual_pos"].values.astype(float)
                arr_q_s_pre = df_calib_q["form_s_pre"].values.astype(float)
                arr_q_w_pre = df_calib_q["form_w_pre"].values.astype(float)
                arr_q_s_cur = df_calib_q["form_s_cur"].values.astype(float)
                arr_q_w_cur = df_calib_q["form_w_cur"].values.astype(float)
                q_event_keys = df_calib_q[["season", "round"]].values
                _, q_event_indices = np.unique(q_event_keys, axis=0, return_inverse=True)
                n_q_unique = len(np.unique(q_event_keys, axis=0))
                q_event_masks = [
                    np.where(q_event_indices == ei)[0] for ei in range(n_q_unique)
                ]

            SQRT2 = float(np.sqrt(2.0))

            def _rank_and_prob_terms(combined_z, actual, masks, sigma, aw):
                """Per-event rank loss + probability calibration terms.

                Returns (mean_spearman, mean_podium_err, mean_pairwise_brier,
                mean_winner_brier, n_scored).  The probability terms use the
                analytic Gaussian-race model that mirrors the Monte Carlo
                simulation: P(i beats j) = Phi((z_j - z_i) / (sigma*sqrt(2))).
                """
                sp_sum = pod_sum = pair_sum = win_sum = 0.0
                n_scored = 0
                for mask in masks:
                    if len(mask) < 3:
                        continue
                    pred = combined_z[mask]
                    act = actual[mask]
                    if np.std(pred) < 1e-9:
                        continue
                    corr, _ = spearmanr(pred, act)
                    if not np.isfinite(corr):
                        continue
                    n_scored += 1
                    sp_sum += corr
                    pred_order = np.argsort(pred)
                    pod_sum += np.mean(np.abs(act[pred_order[:3]] - np.arange(1, 4)))

                    # Pairwise win probabilities under the simulation's noise model
                    diff = (pred[None, :] - pred[:, None]) / (sigma * SQRT2)
                    P = ndtr(diff)  # P[i, j] = P(i finishes ahead of j)
                    Y = (act[:, None] < act[None, :]).astype(float)
                    iu = np.triu_indices(len(mask), k=1)
                    pair_sum += float(np.mean((P[iu] - Y[iu]) ** 2))

                    # Winner probability: analytic "beats everyone" approximation
                    # blended with the Plackett-Luce softmax via aw — the exact
                    # blend production applies to p_win.
                    logP = np.log(np.maximum(P, 1e-12))
                    np.fill_diagonal(logP, 0.0)
                    p_norm = np.exp(logP.sum(axis=1))
                    p_norm_sum = p_norm.sum()
                    p_norm = p_norm / p_norm_sum if p_norm_sum > 0 else np.full(len(mask), 1.0 / len(mask))
                    s_pl = -pred
                    s_pl = s_pl - s_pl.max()  # stable softmax(-pred)
                    e_pl = np.exp(s_pl)
                    p_pl = e_pl / e_pl.sum()
                    p_win = (1.0 - aw) * p_norm + aw * p_pl
                    y_win = (act == act.min()).astype(float)
                    win_sum += float(np.mean((p_win - y_win) ** 2))

                if n_scored == 0:
                    return None
                return (sp_sum / n_scored, pod_sum / n_scored,
                        pair_sum / n_scored, win_sum / n_scored, n_scored)

            def objective(weights):
                """Combined objective over race AND qualifying sessions.

                Every parameter in the vector receives gradient signal:
                  - blending/season/grid/ensemble params -> per-event Spearman +
                    podium terms (race and qualifying), via the same formulas
                    production uses (ratio-form season boost, anchor-delta grid);
                  - noise_factor/min_noise/analytical_win_weight -> analytic
                    pairwise & winner Brier terms mirroring the MC simulation;
                  - DNF alpha/beta/driver/team weights -> Brier on actual DNFs
                    using the exact estimate_dnf_probabilities formula.
                """
                # Unpack race/blending params
                wb_gbm = max(0, weights[0])
                wb_form = max(0, weights[1])
                wb_tm = max(0, weights[2])
                # Production clamps dynamic grid stickiness to [0.4, 0.95]
                wb_grid = np.clip(weights[3], 0.4, 0.95)

                we_pace = max(0, weights[4])
                we_elo = max(0, weights[5])
                we_bt = max(0, weights[6])
                we_mixed = max(0, weights[7])

                w_season = max(1.0, weights[8])
                w_q_season = max(1.0, weights[9])
                w_quali = np.clip(weights[10], 0.0, 1.0)
                aw = np.clip(weights[11], 0.0, 1.0)

                dnf_alpha = max(0.1, weights[12])
                dnf_beta = max(0.1, weights[13])
                dnf_dw = np.clip(weights[14], 0.0, 1.0)
                dnf_tw = np.clip(weights[15], 0.0, 1.0)

                nf = max(0.01, weights[16])
                mn = max(0.01, weights[17])

                # Qualifying ensemble weights (params 18-21)
                we_q_pace = max(0, weights[18]) if len(weights) > 18 else PARAM_DEFAULTS[18]
                we_q_elo = max(0, weights[19]) if len(weights) > 19 else PARAM_DEFAULTS[19]
                we_q_bt = max(0, weights[20]) if len(weights) > 20 else PARAM_DEFAULTS[20]
                we_q_mixed = max(0, weights[21]) if len(weights) > 21 else PARAM_DEFAULTS[21]

                # Noise sigma exactly as simulate_grid derives it for the
                # z-normalised combined pace it receives (std == 1).
                sigma = max(nf * NOISE_STD_MULTIPLIER, mn)

                # ---- Race objective ----
                # Exact production form: weighted average with the current-season
                # boost as a multiplicative weight (a ratio in b, not additive).
                denom_form = arr_w_pre + w_season * arr_w_cur
                form_idx = np.where(
                    denom_form > 1e-9,
                    (arr_s_pre + w_season * arr_s_cur) / np.maximum(denom_form, 1e-9),
                    0.0,
                )
                form_pace = -form_idx  # lower = faster

                # Production baseline: base = -form - w_team * team_form,
                # then pace = w_gbm * gbm + w_base * base.
                base = form_pace + wb_tm * arr_base_team
                raw_pace = wb_gbm * arr_gbm_raw + wb_form * base

                p_z = _event_z(raw_pace, event_indices, n_unique)

                # Current-weekend qualifying blend (mirrors models.py): for drivers
                # with a current-quali result, blend their z-scored pace with the
                # z-scored qualifying position using current_quali_factor (param 10).
                # Applied before grid stickiness, exactly as in production.
                p_z = np.where(
                    quali_blend_mask,
                    (1.0 - w_quali) * p_z + w_quali * q_z_precomputed,
                    p_z,
                )

                # Grid anchor-delta, the actual production formula in models.py:
                # final = grid + pace_z * (1 - stickiness) * MAX_DELTA
                pace_pos = grid_filled + p_z * (1.0 - wb_grid) * 10.0
                a_z = _event_z(pace_pos, event_indices, n_unique)

                # Normalise race ensemble weights
                race_wsum = we_pace + we_elo + we_bt + we_mixed
                if race_wsum < 1e-9:
                    race_wsum = 1.0
                combined_race = (
                    (we_pace / race_wsum) * a_z
                    + (we_elo / race_wsum) * arr_elo
                    + (we_bt / race_wsum) * arr_bt
                    + (we_mixed / race_wsum) * arr_mixed
                )
                # combine_pace z-normalises its output before simulation; mirror
                # that so sigma applies to a std-1 scale.
                combined_race = _event_z(combined_race, event_indices, n_unique)

                race_terms = _rank_and_prob_terms(
                    combined_race, arr_actual_pos, event_masks, sigma, aw,
                )
                if race_terms is None:
                    return 1e6
                mean_sp, mean_pod, mean_pair, mean_win, _ = race_terms

                # DNF Brier — exact estimate_dnf_probabilities formula
                p_global = (arr_dnf_gk + dnf_alpha) / np.maximum(arr_dnf_gn + dnf_alpha + dnf_beta, 1e-9)
                p_drv = (arr_dnf_drv_k + dnf_alpha) / (arr_dnf_drv_n + dnf_alpha + dnf_beta)
                p_drv = np.where(arr_dnf_drv_n > 0, p_drv, p_global)
                p_team = (arr_dnf_team_k + dnf_alpha) / (arr_dnf_team_n + dnf_alpha + dnf_beta)
                p_team = np.where(arr_dnf_team_n > 0, p_team, p_global)
                dnf_wsum = dnf_dw + dnf_tw
                if dnf_wsum > 1e-9:
                    p_dnf = (dnf_dw * p_drv + dnf_tw * p_team) / dnf_wsum
                else:
                    p_dnf = 0.5 * p_drv + 0.5 * p_team
                p_dnf = np.clip(p_dnf * arr_dnf_cmod, dnf_clip_min, dnf_clip_max)
                dnf_brier = float(np.mean((p_dnf - arr_actual_dnf) ** 2))

                race_loss = (-mean_sp + 0.02 * mean_pod
                             + 1.0 * mean_pair + 1.0 * mean_win + 1.0 * dnf_brier)

                # ---- Qualifying objective ----
                quali_loss = None
                if has_quali_data:
                    q_denom = arr_q_w_pre + w_q_season * arr_q_w_cur
                    q_form_idx = np.where(
                        q_denom > 1e-9,
                        (arr_q_s_pre + w_q_season * arr_q_s_cur) / np.maximum(q_denom, 1e-9),
                        0.0,
                    )
                    q_base = -q_form_idx + wb_tm * arr_q_team
                    raw_q = wb_gbm * arr_q_gbm + wb_form * q_base
                    q_z = _event_z(raw_q, q_event_indices, n_q_unique)

                    q_wsum = we_q_pace + we_q_elo + we_q_bt + we_q_mixed
                    if q_wsum < 1e-9:
                        q_wsum = 1.0
                    combined_quali = (
                        (we_q_pace / q_wsum) * q_z
                        + (we_q_elo / q_wsum) * arr_q_elo
                        + (we_q_bt / q_wsum) * arr_q_bt
                        + (we_q_mixed / q_wsum) * arr_q_mixed
                    )
                    combined_quali = _event_z(combined_quali, q_event_indices, n_q_unique)

                    quali_terms = _rank_and_prob_terms(
                        combined_quali, arr_q_actual, q_event_masks, sigma, aw,
                    )
                    if quali_terms is not None:
                        q_sp, q_pod, q_pair, q_win, _ = quali_terms
                        quali_loss = -q_sp + 0.02 * q_pod + 1.0 * q_pair + 1.0 * q_win

                # Combined loss: equal weighting of race and qualifying performance
                if quali_loss is not None:
                    total_loss = 0.5 * race_loss + 0.5 * quali_loss
                else:
                    total_loss = race_loss

                # Regularisation: very light L2 tiebreaker toward defaults.  The
                # bounds (PARAM_BOUNDS) are the primary guardrails; every param now
                # has real gradient signal so no per-block anchoring is needed.
                defaults_arr = np.array(PARAM_DEFAULTS, dtype=float)
                scales = np.maximum(np.abs(defaults_arr), 1.0)
                diffs = ((weights - defaults_arr) / scales) ** 2
                reg = float(np.sum(diffs)) * 1e-6

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
            self.current_weights["objective_score"] = float(best_res.fun)

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
            w_gbm_quali=ens.get("w_gbm_quali", PARAM_DEFAULTS[18]),
            w_elo_quali=ens.get("w_elo_quali", PARAM_DEFAULTS[19]),
            w_bt_quali=ens.get("w_bt_quali", PARAM_DEFAULTS[20]),
            w_mixed_quali=ens.get("w_mixed_quali", PARAM_DEFAULTS[21]),
        )
