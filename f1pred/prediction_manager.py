"""Background prediction manager for continuous async predictions.

Runs predictions periodically in a background thread, detects changes in
input variables, computes position diffs, and broadcasts updates to
connected SSE clients.
"""
from __future__ import annotations

import asyncio
import hashlib
import json
import os
import threading
import time
import httpx
from dataclasses import dataclass, field, asdict
from datetime import datetime, timezone
from typing import Any, Callable, Dict, List, Optional, Set

from .util import get_logger

logger = get_logger(__name__)

__all__ = [
    "PredictionManager",
    "PredictionDiff",
    "DriverMovement",
]

# Human-readable reasons for variable changes
_CHANGE_REASONS = {
    "weather": "Weather forecast updated",
    "grid": "Grid positions changed",
    "calibration": "Model calibration weights updated",
    "roster": "Driver roster changed",
    "features": "Driver form/stats updated",
}


@dataclass
class DriverMovement:
    """A single driver's position change between prediction runs."""
    driver_id: str
    driver_name: str
    code: str
    team: str
    old_position: int
    new_position: int
    direction: int  # positive = moved up (improved), negative = moved down
    reasons: List[str] = field(default_factory=list)

    def to_dict(self) -> Dict[str, Any]:
        return asdict(self)


@dataclass
class PredictionDiff:
    """Diff between two prediction runs for a single session."""
    session: str
    movements: List[DriverMovement] = field(default_factory=list)
    changed_variables: List[str] = field(default_factory=list)
    timestamp: str = ""

    def __post_init__(self):
        if not self.timestamp:
            self.timestamp = datetime.now(timezone.utc).isoformat()

    def to_dict(self) -> Dict[str, Any]:
        return {
            "session": self.session,
            "movements": [m.to_dict() for m in self.movements],
            "changed_variables": self.changed_variables,
            "timestamp": self.timestamp,
        }


def compute_prediction_diff(
    session: str,
    old_predictions: List[Dict[str, Any]],
    new_predictions: List[Dict[str, Any]],
    old_weather: Optional[Dict[str, Any]] = None,
    new_weather: Optional[Dict[str, Any]] = None,
) -> Optional[PredictionDiff]:
    """Compare two prediction result lists and produce a diff.

    Each prediction list is a list of dicts with at least:
      driverId, name, code, constructorName, predicted_position

    Returns None if there are no position changes.
    """
    if not old_predictions or not new_predictions:
        return None

    # Build position maps: driverId -> predicted_position
    old_pos = {
        p["driverId"]: int(p["predicted_position"])
        for p in old_predictions
        if p.get("driverId") and p.get("predicted_position") is not None
    }
    new_pos = {
        p["driverId"]: int(p["predicted_position"])
        for p in new_predictions
        if p.get("driverId") and p.get("predicted_position") is not None
    }

    # Build driver info lookup from new predictions
    driver_info = {}
    for p in new_predictions:
        did = p.get("driverId")
        if did:
            driver_info[did] = {
                "name": p.get("name", ""),
                "code": p.get("code", ""),
                "team": p.get("constructorName", ""),
            }

    # Detect changed variables
    changed_vars: List[str] = []

    # Weather change detection (uses tolerance)
    if old_weather and new_weather:
        weather_keys = ["temp_mean", "rain_sum", "wind_mean", "pressure_mean", "humidity_mean"]
        for wk in weather_keys:
            ov = old_weather.get(wk)
            nv = new_weather.get(wk)
            if ov is not None and nv is not None:
                if abs(ov - nv) > max(0.5, abs(ov) * 0.05):
                    changed_vars.append(_CHANGE_REASONS["weather"])
                    break
    elif (old_weather is None) != (new_weather is None):
        changed_vars.append(_CHANGE_REASONS["weather"])

    # Grid change detection — only flag if a grid value meaningfully changed
    # (i.e. None/missing → integer, or integer changed value)
    def _normalise_grid(v):
        """Normalise grid values: treat None, NaN, 0, and missing as 'unset'."""
        if v is None:
            return None
        if isinstance(v, float):
            import math
            if math.isnan(v) or v == 0.0:
                return None
            return int(v)
        if isinstance(v, int) and v == 0:
            return None
        return int(v) if isinstance(v, (int, float)) else None

    old_grids = {p["driverId"]: _normalise_grid(p.get("grid"))
                 for p in old_predictions if p.get("driverId")}
    new_grids = {p["driverId"]: _normalise_grid(p.get("grid"))
                 for p in new_predictions if p.get("driverId")}
    # Only flag if at least one driver got a *new* grid value (None→int)
    grid_changed = False
    for did in set(old_grids) | set(new_grids):
        og = old_grids.get(did)
        ng = new_grids.get(did)
        if og != ng and ng is not None:
            grid_changed = True
            break
    if grid_changed:
        changed_vars.append(_CHANGE_REASONS["grid"])

    # Feature-level change detection — use tolerance for floats
    def _values_close(a, b, tol=1e-3):
        """Check if two values are effectively equal, handling None/NaN."""
        if a is None and b is None:
            return True
        if a is None or b is None:
            return False
        if isinstance(a, (int, float)) and isinstance(b, (int, float)):
            import math
            if math.isnan(a) and math.isnan(b):
                return True
            if math.isnan(a) or math.isnan(b):
                return False
            return abs(a - b) < tol
        return a == b

    feature_keys = ["form_index", "qualifying_form_index", "team_form_index",
                     "circuit_avg_pos"]
    features_changed = False
    for p_old in old_predictions:
        did = p_old.get("driverId")
        if not did:
            continue
        p_new = next((p for p in new_predictions if p.get("driverId") == did), None)
        if not p_new:
            continue
        for fk in feature_keys:
            if not _values_close(p_old.get(fk), p_new.get(fk)):
                features_changed = True
                break
        if features_changed:
            break
    if features_changed:
        changed_vars.append(_CHANGE_REASONS["features"])

    if not changed_vars:
        changed_vars.append("Prediction model re-evaluated")

    # Compute movements
    movements: List[DriverMovement] = []
    all_drivers = set(old_pos.keys()) | set(new_pos.keys())

    for did in all_drivers:
        op = old_pos.get(did)
        np_ = new_pos.get(did)
        if op is None or np_ is None:
            continue
        if op != np_:
            # direction: positive means moved up (lower position number = better)
            direction = op - np_
            info = driver_info.get(did, {})

            # Build per-driver reasons
            reasons: List[str] = []

            # Check SHAP changes for this driver
            old_shap = next((p.get("shap_values") for p in old_predictions
                           if p.get("driverId") == did), None)
            new_shap = next((p.get("shap_values") for p in new_predictions
                           if p.get("driverId") == did), None)
            if old_shap and new_shap and isinstance(old_shap, dict) and isinstance(new_shap, dict):
                # Find features with largest change
                all_feats_keys = set(old_shap.keys()) | set(new_shap.keys())
                deltas = []
                for fk in all_feats_keys:
                    ov = old_shap.get(fk, 0)
                    nv = new_shap.get(fk, 0)
                    if isinstance(ov, (int, float)) and isinstance(nv, (int, float)):
                        delta = nv - ov
                        if abs(delta) > 0.01:
                            deltas.append((fk, delta))
                deltas.sort(key=lambda x: abs(x[1]), reverse=True)
                for fk, delta in deltas[:3]:
                    label = fk.replace("_", " ").title()
                    dir_word = "improved" if delta < 0 else "worsened"
                    reasons.append(f"{label} {dir_word}")

            if not reasons:
                reasons = list(changed_vars)

            movements.append(DriverMovement(
                driver_id=did,
                driver_name=info.get("name", ""),
                code=info.get("code", ""),
                team=info.get("team", ""),
                old_position=op,
                new_position=np_,
                direction=direction,
                reasons=reasons,
            ))

    if not movements:
        return None

    # Sort by absolute movement magnitude (largest movers first)
    movements.sort(key=lambda m: abs(m.direction), reverse=True)

    return PredictionDiff(
        session=session,
        movements=movements,
        changed_variables=changed_vars,
    )


def _fingerprint_predictions(predictions: List[Dict[str, Any]]) -> str:
    """Generate a stable hash from prediction results for change detection.

    Uses coarse rounding (2 decimal places) to avoid false diffs from
    Monte Carlo probability jitter at rounding boundaries.
    """
    key_data = []
    for p in sorted(predictions, key=lambda x: x.get("driverId", "") or ""):
        key_data.append({
            "driverId": p.get("driverId"),
            "predicted_position": p.get("predicted_position"),
            "p_win": round(p.get("p_win") or 0, 2),
            "p_top3": round(p.get("p_top3") or 0, 2),
            "mean_pos": round(p.get("mean_pos") or 0, 1),
        })
    return hashlib.sha256(json.dumps(key_data, sort_keys=True).encode()).hexdigest()


class PredictionManager:
    """Manages background prediction lifecycle with change detection.

    Runs predictions periodically for the current "next" event,
    detects when results change, and notifies connected clients.
    """

    def __init__(self, cfg, poll_interval: int = 3600, settings_getter: Optional[Callable[[str], Optional[str]]] = None):
        self.cfg = cfg
        self.poll_interval = max(60, poll_interval)  # Minimum 60 seconds
        self._settings_getter = settings_getter
        self._running = False
        self._thread: Optional[threading.Thread] = None
        self._lock = threading.Lock()

        # Latest state
        self._latest_results: Optional[Dict[str, Any]] = None
        self._cache_file = os.path.join(cfg.paths.cache_dir, "latest_predictions.json")

        # Load from disk cache if exists
        if os.path.exists(self._cache_file):
            try:
                with open(self._cache_file, "r") as f:
                    self._latest_results = json.load(f)
                logger.info("[PredictionManager] Loaded %d rounds from disk cache",
                            len(self._latest_results.get("rounds", {})))
            except Exception as e:
                logger.warning("[PredictionManager] Failed to load disk cache: %s", e)

        self._latest_diffs: List[PredictionDiff] = []
        self._previous_fingerprints: Dict[str, str] = {}  # session -> hash
        self._previous_predictions: Dict[str, List[Dict[str, Any]]] = {}  # session -> predictions
        self._previous_weather: Dict[str, Dict[str, Any]] = {}  # session -> weather
        self._last_update: Optional[str] = None
        self._status: str = "idle"  # idle, running, error

        # SSE subscriber management
        self._subscribers: Set[asyncio.Queue] = set()
        self._subscriber_lock = threading.Lock()

    @property
    def latest_results(self) -> Optional[Dict[str, Any]]:
        with self._lock:
            return self._latest_results

    @property
    def latest_diffs(self) -> List[Dict[str, Any]]:
        with self._lock:
            return [d.to_dict() for d in self._latest_diffs[-20:]]

    @property
    def last_update(self) -> Optional[str]:
        with self._lock:
            return self._last_update

    @property
    def status(self) -> str:
        with self._lock:
            return self._status

    def subscribe(self) -> asyncio.Queue:
        """Register a new SSE subscriber. Returns a queue for receiving events."""
        q: asyncio.Queue = asyncio.Queue(maxsize=50)
        with self._subscriber_lock:
            self._subscribers.add(q)
        logger.info("[PredictionManager] New subscriber (total: %d)", len(self._subscribers))
        return q

    def unsubscribe(self, q: asyncio.Queue) -> None:
        """Remove an SSE subscriber."""
        with self._subscriber_lock:
            self._subscribers.discard(q)
        logger.info("[PredictionManager] Subscriber removed (total: %d)", len(self._subscribers))

    def _broadcast(self, event: Dict[str, Any]) -> None:
        """Send an event to all connected subscribers."""
        with self._subscriber_lock:
            dead: List[asyncio.Queue] = []
            for q in self._subscribers:
                try:
                    q.put_nowait(event)
                except asyncio.QueueFull:
                    dead.append(q)
                except Exception:
                    dead.append(q)
            for q in dead:
                self._subscribers.discard(q)

    def start(self) -> None:
        """Start the background prediction loop."""
        if self._running:
            return
        self._running = True
        self._thread = threading.Thread(target=self._run_loop, daemon=True, name="PredictionManager")
        self._thread.start()
        logger.info("[PredictionManager] Started (poll_interval=%ds)", self.poll_interval)

    def stop(self) -> None:
        """Stop the background prediction loop."""
        self._running = False
        if self._thread:
            self._thread.join(timeout=10)
            self._thread = None
        logger.info("[PredictionManager] Stopped")

    def _run_loop(self) -> None:
        """Main background loop: resolve next event, run predictions, detect changes."""
        # Brief startup delay to let the server finish initializing
        time.sleep(1)
        cycle_count = 0

        while self._running:
            try:
                self._run_season_cycle(cycle_count)
            except Exception as e:
                logger.warning("[PredictionManager] Prediction cycle failed: %s", e)
                with self._lock:
                    self._status = "error"
                self._broadcast({"type": "status", "status": "error", "message": str(e)})

            cycle_count += 1
            # Sleep in small increments so we can stop quickly
            for _ in range(self.poll_interval):
                if not self._running:
                    break
                time.sleep(1)

    def _run_season_cycle(self, cycle_count: int) -> None:
        """Orchestrate predicting all rounds across the season."""
        from .data.jolpica import JolpicaClient
        from .predict import resolve_event

        with self._lock:
            self._status = "running"
        self._broadcast({"type": "status", "status": "running"})

        jc = JolpicaClient(
            self.cfg.data_sources.jolpica.base_url,
            self.cfg.data_sources.jolpica.timeout_seconds,
            self.cfg.data_sources.jolpica.rate_limit_sleep,
        )

        try:
            curr_s, next_r, _ = resolve_event(jc, "current", "next")
        except Exception as e:
            logger.warning("[PredictionManager] Failed to resolve next event: %s", e)
            curr_s, next_r = None, None

        try:
            s_str = str(curr_s) if curr_s else "current"
            schedule = jc.get_season_schedule(s_str)
            # Prioritize next round: sort schedule so next_r comes first
            schedule = sorted(
                [r for r in schedule if r.get("round")],
                key=lambda r: 0 if int(r["round"]) == next_r else int(r["round"])
            )
        except Exception as e:
            logger.warning("[PredictionManager] Failed to fetch schedule: %s", e)
            with self._lock:
                self._status = "error"
            return
            
        season = curr_s if curr_s else schedule[0].get("season")
        
        # Initialize the global season structure if needed
        with self._lock:
            if not self._latest_results or self._latest_results.get("season") != season:
                self._latest_results = {"season": season, "rounds": {}}

        # Process each round
        for r_info in schedule:
            if not self._running:
                break
                
            round_i = int(r_info["round"])
            
            # Determine if we should poll this round
            is_next = (round_i == next_r)

            with self._lock:
                existing_data = self._latest_results["rounds"].get(str(round_i))

            # Polling strategy:
            # 1. Always poll the 'next' round.
            # 2. Poll any round that exists but is not 'frozen' (i.e., results not yet finalized).
            #    We only consider competitive sessions (race/qual/sprint) for completion.
            # 3. For finished & frozen rounds, only re-poll once a day (every 24 cycles).
            should_poll = is_next
            if not should_poll:
                if existing_data:
                    # Only check completion for competitive sessions
                    competitive = ["race", "qualifying", "sprint", "sprint_qualifying"]
                    sessions_to_check = {
                        k: v for k, v in existing_data.get("sessions", {}).items()
                        if k in competitive
                    }
                    is_complete = all(s.get("frozen", False) for s in sessions_to_check.values())

                    if not is_complete:
                        should_poll = True
                    elif cycle_count % 24 == 0:
                        should_poll = True
                else:
                    # No data yet for this round, poll it
                    should_poll = True

            if should_poll:
                self._predict_round(jc, season, round_i, r_info)
            
        now = datetime.now(timezone.utc).isoformat()
        with self._lock:
            self._status = "idle"
            self._last_update = now
            self._save_to_disk()
            
        self._broadcast({"type": "status", "status": "idle", "timestamp": now})

    def _save_to_disk(self) -> None:
        """Persist current predictions to disk."""
        if not self._latest_results:
            return
        try:
            # Ensure directory exists
            os.makedirs(os.path.dirname(self._cache_file), exist_ok=True)
            with open(self._cache_file, "w") as f:
                json.dump(self._latest_results, f)
        except Exception as e:
            logger.warning("[PredictionManager] Failed to save disk cache: %s", e)

    def _get_setting(self, key: str) -> Optional[str]:
        if not self._settings_getter:
            return None
        try:
            value = self._settings_getter(key)
        except Exception as e:
            logger.warning("[PredictionManager] Failed to read setting '%s': %s", key, e)
            return None
        if value is None:
            return None
        if isinstance(value, str):
            value = value.strip()
        return value or None

    def _send_discord_webhook(self, diff: PredictionDiff, event_title: str) -> None:
        webhook_url = self._get_setting("discord_webhook_url")
        if not webhook_url:
            return

        movement_lines = []
        for movement in diff.movements[:8]:
            direction = "up" if movement.direction > 0 else "down"
            movement_lines.append(
                f"- {movement.code or movement.driver_name}: P{movement.old_position} -> P{movement.new_position} ({direction} {abs(movement.direction)})"
            )

        content_lines = [
            "F1 Predictor update",
            f"Event: {event_title}",
            f"Session: {diff.session}",
            "",
            "Position changes:",
            *movement_lines,
        ]

        if diff.changed_variables:
            content_lines.extend([
                "",
                "Detected input changes:",
                "- " + " | ".join(diff.changed_variables[:4]),
            ])

        payload = {"content": "\n".join(content_lines)}

        delay_s = 1.0
        max_attempts = 3
        for attempt in range(1, max_attempts + 1):
            try:
                response = httpx.post(webhook_url, json=payload, timeout=10.0)
                if response.status_code in (200, 204):
                    return

                if response.status_code == 429 and attempt < max_attempts:
                    retry_after = response.headers.get("Retry-After")
                    try:
                        delay_s = max(float(retry_after), delay_s)
                    except (TypeError, ValueError):
                        pass
                    logger.warning("[PredictionManager] Discord webhook rate limited; retrying in %.1fs", delay_s)
                    time.sleep(delay_s)
                    delay_s *= 2
                    continue

                if attempt < max_attempts:
                    logger.warning(
                        "[PredictionManager] Discord webhook failed (status=%s), retrying in %.1fs",
                        response.status_code,
                        delay_s,
                    )
                    time.sleep(delay_s)
                    delay_s *= 2
                    continue

                logger.warning(
                    "[PredictionManager] Discord webhook failed after retries (status=%s): %s",
                    response.status_code,
                    response.text[:300],
                )
            except Exception as e:
                if attempt < max_attempts:
                    logger.warning(
                        "[PredictionManager] Discord webhook error (%s), retrying in %.1fs",
                        e,
                        delay_s,
                    )
                    time.sleep(delay_s)
                    delay_s *= 2
                    continue
                logger.warning("[PredictionManager] Discord webhook failed after retries: %s", e)

    def _predict_round(self, jc, season_i, round_i, race_info) -> None:
        import math
        from .predict import run_predictions_for_event
        
        event_title = f"{race_info.get('raceName', 'Event')} {season_i} R{round_i}"
        logger.info("[PredictionManager] Running predictions for %s", event_title)

        all_sessions = self.cfg.modelling.targets.session_types
        sessions = []
        for s in all_sessions:
            if s in ("qualifying", "race"):
                sessions.append(s)
            elif s == "sprint" and "Sprint" in race_info:
                sessions.append(s)
            elif s == "sprint_qualifying" and "SprintQualifying" in race_info:
                sessions.append(s)
        if not sessions:
            sessions = ["qualifying", "race"]

        def progress_cb(msg: str):
            self._broadcast({"type": "log", "message": msg})

        results = run_predictions_for_event(
            self.cfg,
            season=str(season_i),
            rnd=str(round_i),
            sessions=sessions,
            return_results=True,
            progress_callback=progress_cb,
            use_actuals=True, # Use actuals for background manager too
        )

        if not results:
            logger.info("[PredictionManager] No results generated for %s", event_title)
            return

        output = {
            "season": results["season"],
            "round": results["round"],
            "event_name": race_info.get("raceName", ""),
            "sessions": {},
        }

        def _sanitize(v):
            if isinstance(v, float) and (math.isnan(v) or math.isinf(v)):
                return None
            if isinstance(v, dict):
                return {k: _sanitize(vv) for k, vv in v.items()}
            if isinstance(v, list):
                return [_sanitize(i) for i in v]
            if hasattr(v, "isoformat"):
                return v.isoformat()
            return v

        all_diffs: List[PredictionDiff] = []

        for sess, data in results["sessions"].items():
            ranked_df = data["ranked"]
            ranked_list = ranked_df.to_dict(orient="records")
            is_frozen = data.get("frozen", False)

            for row in ranked_list:
                for k, v in row.items():
                    row[k] = _sanitize(v)

            weather = data.get("meta", {}).get("weather", {})
            output["sessions"][sess] = {
                "predictions": ranked_list,
                "weather": weather,
                "frozen": is_frozen
            }

            cache_key = f"{round_i}_{sess}"
            new_fp = _fingerprint_predictions(ranked_list)
            old_fp = self._previous_fingerprints.get(cache_key)

            if old_fp is not None and old_fp != new_fp:
                old_preds = self._previous_predictions.get(cache_key, [])
                old_weather = self._previous_weather.get(cache_key)
                diff = compute_prediction_diff(
                    session=f"R{round_i}_{sess}",
                    old_predictions=old_preds,
                    new_predictions=ranked_list,
                    old_weather=old_weather,
                    new_weather=weather,
                )
                if diff:
                    all_diffs.append(diff)

            self._previous_fingerprints[cache_key] = new_fp
            self._previous_predictions[cache_key] = ranked_list
            self._previous_weather[cache_key] = weather

        now = datetime.now(timezone.utc).isoformat()
        with self._lock:
            if not self._latest_results:
                self._latest_results = {"season": season_i, "rounds": {}}
            self._latest_results["rounds"][str(round_i)] = output
            
            self._latest_diffs.extend(all_diffs)
            if len(self._latest_diffs) > 50:
                self._latest_diffs = self._latest_diffs[-50:]
            self._last_update = now
            self._save_to_disk()

        for diff in all_diffs:
            self._broadcast({
                "type": "diff",
                "data": diff.to_dict(),
                "timestamp": now,
            })
            self._send_discord_webhook(diff, event_title)
            
        self._broadcast({
            "type": "prediction_round",
            "data": output,
            "timestamp": now,
        })
