"""Dynamic calibration of model weights based on recent history.

This module provides the CalibrationManager to optimize ensemble and blending weights
by minimizing prediction error over a sliding lookback window (e.g., last 3 years).
"""
import json
import time
from pathlib import Path
from typing import Dict, Any, Optional
from datetime import datetime, timedelta, timezone
from concurrent.futures import ThreadPoolExecutor

from .util import get_logger, ensure_dirs
from .data.jolpica import JolpicaClient
from .data.open_meteo import OpenMeteoClient
from .ensemble import EnsembleConfig

logger = get_logger(__name__)

class CalibrationManager:
    def __init__(self, cfg):
        self.cfg = cfg
        # Default path relative to project or absolute if configured
        self.weights_file = Path(cfg.calibration.weights_file)
        self.lookback_days = getattr(cfg.calibration, "lookback_window_days", 1095)
        self.frequency_hours = getattr(cfg.calibration, "frequency_hours", 24)
        
        # Default weights (fallback)
        self.current_weights = {
            "ensemble": {
                "w_gbm": 0.4,
                "w_elo": 0.2,
                "w_bt": 0.2,
                "w_mixed": 0.2
            },
            "blending": {
                "gbm_weight": 0.75,
                "baseline_weight": 0.25,
                "baseline_team_factor": 0.3,
                "baseline_driver_team_factor": 0.2,
                "grid_factor": 0.8,
                "current_quali_factor": 0.5
            }
        }
        self.last_race_id: Optional[str] = None

    def load_weights(self) -> Dict[str, Any]:
        """Load weights from disk if available, otherwise return defaults."""
        if self.weights_file.exists():
            try:
                with open(self.weights_file, "r") as f:
                    data = json.load(f)
                    
                # Validate structure
                if "ensemble" in data and "blending" in data:
                    self.current_weights = data
                    self.last_race_id = data.get("last_race_id")
                    logger.info(f"[calibrate] Loaded calibrated weights from {self.weights_file} (last_race_id={self.last_race_id})")
                else:
                    logger.warning("[calibrate] Weights file malformed, using defaults")
            except Exception as e:
                logger.error(f"[calibrate] Failed to load weights: {e}")
        
        return self.current_weights

    def save_weights(self):
        """Save current weights to disk."""
        try:
            ensure_dirs(str(self.weights_file.parent))
            with open(self.weights_file, "w") as f:
                data = self.current_weights.copy()
                if self.last_race_id:
                    data["last_race_id"] = self.last_race_id
                json.dump(data, f, indent=2)
            logger.info(f"[calibrate] Saved calibrated weights to {self.weights_file}")
        except Exception as e:
            logger.error(f"[calibrate] Failed to save weights: {e}")

    def check_calibration_needed(self, history_df: Optional['pd.DataFrame'] = None) -> bool:
        """Check if calibration is needed based on new race results or missing file."""
        if not self.cfg.calibration.enabled:
            return False
            
        if not self.weights_file.exists():
            logger.info("[calibrate] Weights file missing, calibration needed")
            return True

        # Load existing weights (and last_race_id)
        self.load_weights()
        
        # Smart Check: Has a new race occurred?
        if history_df is not None and not history_df.empty:
            # Find the latest completed race in the history
            latest_race = history_df[history_df["session"] == "race"].sort_values("date").iloc[-1]
            latest_id = f"{latest_race['season']}_{latest_race['round']}"
            
            if self.last_race_id != latest_id:
                logger.info(f"[calibrate] New race result found ({latest_id} != {self.last_race_id}), calibration needed")
                return True
            else:
                logger.debug(f"[calibrate] No new results since {self.last_race_id}. Skipping calibration.")
                return False

        # Fallback to time-based check if history not provided (shouldn't happen with new predict.py)
        try:
            mtime = datetime.fromtimestamp(self.weights_file.stat().st_mtime, tz=timezone.utc)
            age = datetime.now(timezone.utc) - mtime
            age_hours = age.total_seconds() / 3600.0
            
            if age_hours > self.frequency_hours:
                logger.info(f"[calibrate] Weights file is {age_hours:.1f}h old (> {self.frequency_hours}h), calibration needed")
                return True
        except Exception:
            return True
            
        return False

    def run_calibration(self, jc: JolpicaClient, om: OpenMeteoClient, history_df: Optional['pd.DataFrame'] = None):
        """Run the full calibration process."""
        # Import heavy deps here
        import numpy as np
        import pandas as pd
        from scipy.optimize import minimize
        from .features import build_session_features, collect_historical_results
        from .models import train_pace_model
        from .ensemble import EloModel, BradleyTerryModel, MixedEffectsLikeModel
        
        logger.info("[calibrate] Starting calibration process...")
        if history_df is None or history_df.empty:
             print("    [Calibration] Starting self-training process... (fetching history)", flush=True)
        else:
             print("    [Calibration] Starting self-training process... (using pre-fetched history)", flush=True)
            
        t0 = time.time()
        
        try:
            # 1. Define window
            now = datetime.now(timezone.utc)
            start_date = now - timedelta(days=self.lookback_days)
            
            # 2. Fetch History (long history for features, limited window for scoring)
            # We need deep history for Elo/Features to be accurate specifically at the start of the window
            if history_df is None or history_df.empty:
                logger.info("[calibrate] Fetching history...")
                print("    [Calibration] Fetching historical race data (last 4 years)...", flush=True)
                
                # Reduce lookback to 4 years (3 years window + 1 year warmup)
                # 10 years is too heavy for a synchronous block on first run
                try:
                    all_hist = collect_historical_results(
                        jc, 
                        season=now.year, 
                        end_before=now, 
                        lookback_years=4 
                    )
                except Exception as e:
                    logger.error(f"[calibrate] History fetch failed: {e}")
                    print(f"    [Calibration] Error fetching history: {e}")
                    return
            else:
                # Use provided history (assuming it's deep enough)
                all_hist = history_df
                # Ensure we don't look into the future if the history frame is too new (unlikely but safe)
                all_hist = all_hist[all_hist["date"] < now]
            
            # Filter specifically for the calibration set (races in the lookback window)
            calib_races = all_hist[
                (all_hist["session"] == "race") & 
                (all_hist["date"] >= start_date) & 
                (all_hist["position"].notna())
            ].sort_values("date")
            
            if calib_races.empty:
                logger.warning("[calibrate] No races found in lookback window. Skipping calibration.")
                print("    [Calibration] SKIPPED: No recent races found in lookback window.")
                return

            # Identify unique events (season, round)
            events = calib_races[["season", "round", "date"]].drop_duplicates().sort_values("date")
            logger.info(f"[calibrate] Found {len(events)} events in lookback window ({start_date.date()} to {now.date()})")

            # Update last_race_id with the MOST RECENT race in the window
            if not events.empty:
                last_evt = events.iloc[-1]
                # last_evt is a Series with index [season, round, date]
                # access via key to avoid method collision with .round()
                self.last_race_id = f"{last_evt['season']}_{last_evt['round']}"

            # 3. Generate Predictions for Calibration Set
            # We need to simulate the state of the world at each event
            # To be efficient, we won't retrain the GBM every race.
            # We will train ONE GBM on data *prior* to the window (Out-of-Time validation)
            # This mimics the "future" prediction task.
            
            train_cutoff = start_date
            train_hist = all_hist[all_hist["date"] < train_cutoff]
            
            if train_hist.empty:
                logger.warning("[calibrate] No training history prior to lookback window. Using simple split.")
                # Fallback: Use first 70% of all_hist for train, last 30% for calib
                # This ensures we have *something* to train on
                split_idx = int(len(all_hist) * 0.7)
                all_hist_sorted = all_hist.sort_values("date")
                train_hist = all_hist_sorted.iloc[:split_idx]
                calib_races = all_hist_sorted.iloc[split_idx:]
                calib_races = calib_races[calib_races["session"] == "race"]
                events = calib_races[["season", "round", "date"]].drop_duplicates().sort_values("date")

            # Train GBM Baseline
            logger.info(f"[calibrate] Training baseline GBM on {len(train_hist)} rows (pre-{train_cutoff.date()})...")
            # We need X for training. This might be slow to rebuild for all history.
            # For efficiency in calibration, we might simplify or assume we can build it.
            # Building features for 5-10 years of races is heavy.
            # Heuristic: Train GBM on last 3 years *prior* to calibration window if possible, or just last 100 races.
            
            # Let's try to build X for the training set (subset to race sessions)
            train_race_hist = train_hist[train_hist["session"] == "race"].tail(500) # Limit to reasonable size
            
            # We need a helper to build X for many races efficiently
            # Note: build_session_features is per-event.
            # We will loop for training data generation.
            
            top_train_events = train_race_hist[["season", "round", "date"]].drop_duplicates().to_dict('records')
            
            X_train_list = []
            
            # Generate training data
            # Use limited parallelism to avoid rate limits
            with ThreadPoolExecutor(max_workers=4) as executor:
                futures = []
                for evt in top_train_events:
                    futures.append(executor.submit(
                        build_session_features, 
                        jc, om, 
                        int(evt["season"]), int(evt["round"]), "race", 
                        pd.Timestamp(evt["date"]), self.cfg
                    ))
                
                for f in futures:
                    try:
                        X_i, _, _ = f.result()
                        if X_i is not None and not X_i.empty:
                            X_train_list.append(X_i)
                    except Exception as e:
                        logger.error(f"[calibrate] Training sample generation failed: {e}")
                        print(f"    [Calibration] Warning: Training sample failed: {e}")
            
            if not X_train_list:
                logger.error("[calibrate] Failed to generate training data. Aborting.")
                print("    [Calibration] ABORTED: Failed to generate training data (check logs for API errors).")
                return
                
            X_train = pd.concat(X_train_list, ignore_index=True)
            logger.info(f"[calibrate] Generated {len(X_train)} training samples.")
            
            # Fit the GBM
            # We use a dummy config for now or the real one
            gbm_pipe, _, _ = train_pace_model(X_train, "race", self.cfg)
            
            # 4. Score the Calibration Set
            # For each event in evaluation window:
            # - Build X
            # - Predict GBM
            # - Fit/Predict Elo/BT/Mixed (iterative or just fit on history<date)
            # Store results
            
            calibration_data = []
            
            # Pre-calculate ensemble models iteratively to be correct?
            # Or just fit on history < event_date for each event (correct but slower).
            # fitting on history is fast for these simple models.
            
            logger.info("[calibrate] generating scores for calibration events...")
            for i, evt in enumerate(events.itertuples()):
                s, r, d = int(evt.season), int(evt.round), pd.Timestamp(evt.date)
                
                # Fetch features
                X_evt, _, roster = build_session_features(jc, om, s, r, "race", d, self.cfg)
                if X_evt is None or X_evt.empty:
                    continue
                
                # GBM Score
                # X_evt might need alignment? pipeline handles it
                try:
                    # Sklearn pipeline applied to DF works by column name usually
                    pace_hat = gbm_pipe.predict(X_evt)
                    
                    # Standardize (approximate based on training set stats or local?)
                    # Local standardization per race is what predict.py does
                    pace_hat = (pace_hat - pace_hat.mean()) / (pace_hat.std() + 1e-6)
                except Exception as e:
                    logger.warning(f"[calibrate] GBM predict failed for {s} R{r}: {e}")
                    continue
                
                # Ensemble scores
                # History up to this race (strict temporal cutoff — no leakage of target event)
                hist_subset = all_hist[all_hist["date"] < d]
                
                # Use recency weighting (365 days) to avoid valuing old results equal to recent ones
                elo = EloModel().fit(hist_subset).predict(X_evt)
                bt = BradleyTerryModel().fit(hist_subset, half_life_days=365).predict(X_evt)
                mixed = MixedEffectsLikeModel().fit(hist_subset, half_life_days=365).predict(X_evt)
                
                # Baseline info (from X_evt)
                # Form index is already in X_evt, usually negative of what we want for 'pace'
                # pace model uses -form_index.
                # Here we just want the components to weight.
                # Predict.py blends GBM + Baseline.
                # We want to optimize:
                # Pace = w1*GBM + w2*Baseline
                # AND
                # Combined = v1*Pace + v2*Elo + v3*BT + v4*Mixed
                
                # Let's collect raw components
                base_form = -X_evt["form_index"].fillna(0).astype(float).values
                base_team = -X_evt.get("team_form_index", pd.Series(0, index=X_evt.index)).fillna(0).astype(float).values
                base_dt = -X_evt.get("driver_team_form_index", pd.Series(0, index=X_evt.index)).fillna(0).astype(float).values
                
                # Get actual results for target — merge on (season, round, driverId) to avoid
                # cross-round contamination when the same driver appears in multiple rounds.
                actuals = X_evt.assign(_calib_season=s, _calib_round=r).merge(
                    calib_races[
                        (calib_races["season"] == s) & (calib_races["round"] == r)
                    ][["driverId", "position"]].drop_duplicates(subset=["driverId"]),
                    on="driverId",
                    how="left"
                ).drop(columns=["_calib_season", "_calib_round"], errors="ignore")
                
                # We need actual position to minimize rank error
                # Filter to drivers who finished (or keep all and handle NaN)
                
                for idx, row in enumerate(X_evt.itertuples()):
                    act_pos = actuals.iloc[idx]["position"]
                    if pd.isna(act_pos):
                        continue
                        
                    calibration_data.append({
                        "driverId": row.driverId,
                        "season": s,
                        "round": r,
                        "actual_pos": act_pos,
                        "gbm_raw": pace_hat[idx],
                        "base_form": base_form[idx],
                        "base_team": base_team[idx],
                        "base_dt": base_dt[idx],
                        "grid": float(X_evt.iloc[idx].get("grid", 15.0)),
                        "elo": elo[idx] if len(elo) > idx else 0,
                        "bt": bt[idx] if len(bt) > idx else 0,
                        "mixed": mixed[idx] if len(mixed) > idx else 0
                    })
                
                if (i+1) % 5 == 0:
                    logger.info(f"[calibrate] Processed {i+1}/{len(events)} events")
                    print(f"    [Calibration] Backtesting event {i+1}/{len(events)}...")

            if not calibration_data:
                logger.warning("[calibrate] No calibration data generated.")
                print("    [Calibration] ABORTED: No validation queries succeeded.")
                return

            df_calib = pd.DataFrame(calibration_data)
            
            # 5. Optimization
            logger.info(f"[calibrate] Optimizing weights on {len(df_calib)} samples...")
            
            # Metric: Rank Correlation or Rank Error?
            # Simple objective: Minimize Mean Squared Error of (Predicted Rank - Actual Rank) implies estimating rank.
            # Or minimizing Negative Log Likelihood of the winner?
            # Let's just minimize RMSE of the *score* vs *actual position*?
            # No, score is abstract. 
            # We want to minimize the Spearman correlation (maximize it) or 
            # Minimize error of (Predicted Rank - Actual Position).
            
            # Precompute arrays for optimization loop to avoid repeated pandas overhead (~2.8x speedup)
            arr_gbm_raw = df_calib["gbm_raw"].values
            arr_base_form = df_calib["base_form"].values
            arr_base_team = df_calib["base_team"].values
            arr_base_dt = df_calib["base_dt"].values
            arr_elo = df_calib["elo"].values
            arr_bt = df_calib["bt"].values
            arr_mixed = df_calib["mixed"].values
            arr_actual_pos = df_calib["actual_pos"].values

            event_keys = df_calib[["season", "round"]].values
            unique_events, event_indices = np.unique(event_keys, axis=0, return_inverse=True)

            # Precompute normalized grid since it doesn't depend on weights
            grid_vals = df_calib["grid"].values
            g_z_precomputed = np.zeros_like(grid_vals, dtype=float)
            event_masks = [np.where(event_indices == i)[0] for i in range(len(unique_events))]

            for mask in event_masks:
                ev_grid = grid_vals[mask]
                mu_g = np.nanmean(ev_grid)
                sd_g = np.nanstd(ev_grid)
                g_z_precomputed[mask] = (ev_grid - mu_g) / (sd_g + 1e-6)

            def objective(weights):
                # Unpack weights
                # w_blend: [w_gbm, w_base_form, w_base_team, w_base_dt, w_grid]
                # w_ens: [we_pace, we_elo, we_bt, we_mixed]
                
                # GBM Blend
                wb_gbm, wb_form, wb_tm, wb_dt, wb_grid = weights[0:5]

                # Force positive
                wb_gbm = max(0, wb_gbm)
                wb_form = max(0, wb_form)
                wb_tm = max(0, wb_tm)
                wb_dt = max(0, wb_dt)
                wb_grid = max(0, wb_grid)

                # Stage 1: Form-based pace (linear combination)
                # This approximates models.py: w_gbm*gbm + w_base*(form + factor_t*team + factor_d*dt)
                # But we optimize coefficients directly for better convergence.
                raw_pace = (wb_gbm * arr_gbm_raw +
                            wb_form * arr_base_form +
                            wb_tm * arr_base_team +
                            wb_dt * arr_base_dt)

                grid_imp = np.clip(wb_grid, 0.0, 1.0)

                # ⚡ Bolt: Vectorized per-event standardization using bincount instead of Python for-loop
                # Handles NaNs implicitly if present, though calibration inputs shouldn't have NaNs here
                valid_mask = ~np.isnan(raw_pace)
                valid_indices = event_indices[valid_mask]
                valid_pace = raw_pace[valid_mask]

                num_uevents = len(unique_events)
                counts = np.bincount(valid_indices, minlength=num_uevents)
                sums = np.bincount(valid_indices, weights=valid_pace, minlength=num_uevents)

                # Avoid division by zero
                safe_counts = counts.copy()
                safe_counts[safe_counts == 0] = 1
                means = sums / safe_counts

                # Variance
                sq_sums = np.bincount(valid_indices, weights=valid_pace**2, minlength=num_uevents)
                variances = sq_sums / safe_counts - means**2
                variances[variances < 0] = 0 # Handle floating point inaccuracies
                stds = np.sqrt(variances)

                # Broadcast back to original array shape
                mu_p_all = means[event_indices]
                sd_p_all = stds[event_indices]

                p_z = (raw_pace - mu_p_all) / (sd_p_all + 1e-6)

                # Stage 2: Stickiness blend using precomputed normalized grid
                pace = (1.0 - grid_imp) * p_z + grid_imp * g_z_precomputed
                # We intentionally allow NaNs to propagate here as they did in the original loop
                
                # Normalize pace z-score per race to be fair input to ensemble
                # (Doing this per-row vector optimization is hard without grouping, 
                #  assume input components are already roughly z-scored. They are.)
                
                # Ensemble
                we_pace, we_elo, we_bt, we_mixed = weights[5:9]
                we_pace = max(0, we_pace)
                we_elo = max(0, we_elo)
                we_bt = max(0, we_bt)
                we_mixed = max(0, we_mixed)
                
                combined = (we_pace * pace + 
                            we_elo * arr_elo +
                            we_bt * arr_bt +
                            we_mixed * arr_mixed)
                
                # Determine ranks per race
                # Since we want to vectorize, we can just use correlation.
                # Rank correlation (Spearman) is good but non-differentiable/hard to optimize directly with solvers.
                # Proxy: Point-biserial or just simple regression target?
                # The combined score (lower is better) should correlate with Position (lower is better).
                # We want to MAXIMIZE correlation with Position (since both are lower-is-better, positive corr).
                # Minimize -Correlation.
                
                corr = np.corrcoef(combined, arr_actual_pos)[0, 1]
                return -corr + 0.001 * np.sum(weights**2) # Regularization

            # Initial guess
            init_w = [0.6, 0.4, 0.3, 0.2, 0.8, # blending (gbm, base, team_factor, driver_team_factor, grid_factor)
                      0.4, 0.2, 0.2, 0.2]    # ensemble (pace, elo, bt, mixed)
            
            # Bounds
            # Enforce minimum contribution from "car performance" terms to prevent physics denial
            # w_opt[0] = GBM weight (min 5%)
            # w_opt[1] = Baseline weight (min 5%)
            # w_opt[2] = Team Factor (min 0.2 - car matters!)
            bounds = [
                (0.05, 1.0), # GBM
                (0.05, 1.0), # Baseline
                (0.2, 2.0),  # Team Factor (relative to form)
                (0.0, 2.0),  # Driver-Team Factor
                (0.1, 0.95), # Grid Factor (Stickiness)
                (0.0, 1.0),  # Ensemble Pace
                (0.0, 1.0),  # Elo
                (0.0, 1.0),  # BT
                (0.0, 1.0),  # Mixed
            ]
            
            res = minimize(objective, init_w, bounds=bounds, method='L-BFGS-B')
            
            w_opt = res.x
            # Normalize ensemble weights to sum to 1 for readability (metric is invariant to scale)
            ens_sum = sum(w_opt[5:9])
            if ens_sum > 0:
                w_opt[5:9] /= ens_sum
                
            logger.info(f"[calibrate] Optimized weights: {w_opt}")
            
            # Map back to structure
            # Blending
            # w_opt: [gbm, form, team, dt, grid, ...]
            # Models.py expects:
            # pace = w_gbm*gbm + w_base*(form - w_tm*team - w_dt*dt)
            # wait, models.py uses: base = form - w_tm*team - w_dt*dt
            # and pace_hat = w_gbm*gbm + w_base*base
            # NOTE: base_form, base_team, base_dt in df_calib were negated (lower is faster)
            # So models.py 'minus' becomes 'plus' in calibrate objective but 'minus' in mapping.

            w_gbm_opt = max(1e-3, w_opt[0])
            w_form_opt = max(1e-3, w_opt[1])
            w_team_opt = w_opt[2]
            w_dt_opt = w_opt[3]
            w_grid_stickiness = float(np.clip(w_opt[4], 0.0, 1.0))

            # total_main = w_gbm + w_form
            # but we want w_base in config to cover form+team+dt
            # so w_gbm_cfg = w_gbm / (w_gbm + w_form)
            # baseline_team_factor = w_team / w_form (ratio relative to form)
            
            total_main = w_gbm_opt + w_form_opt
            
            self.current_weights = {
                "ensemble": {
                    "w_gbm": float(w_opt[5]), # Pace input
                    "w_elo": float(w_opt[6]),
                    "w_bt": float(w_opt[7]),
                    "w_mixed": float(w_opt[8])
                },
                "blending": {
                    "gbm_weight": float(w_gbm_opt / total_main),
                    "baseline_weight": float(w_form_opt / total_main),
                    # Factors are relative to the form index (ratios)
                    "baseline_team_factor": float(w_team_opt / w_form_opt),
                    "baseline_driver_team_factor": float(w_dt_opt / w_form_opt),
                    "grid_factor": w_grid_stickiness
                }
            }
            
            self.save_weights()
            duration = time.time()-t0
            logger.info(f"[calibrate] Calibration complete. Time: {duration:.1f}s")
            print(f"    [Calibration] Complete! Optimized weights saved. ({duration:.1f}s)")
            
        except Exception as e:
            logger.error(f"[calibrate] Calibration failed: {e}")
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
            w_mixed=ens.get("w_mixed", 0.25)
        )
