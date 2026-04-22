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
from typing import Any, Dict, List, Optional, Set

from .util import get_logger

logger = get_logger(__name__)

__all__ = [
    "PredictionManager",
    "PredictionDiff",
    "DriverMovement",
]

# Human-readable reasons for variable changes
_CHANGE_REASONS = {
    "weather": "🌦️ Weather forecast updated",
    "grid": "🚥 Grid positions changed",
    "calibration": "⚖️ Model calibration weights updated",
    "roster": "👥 Driver roster changed",
    "features": "📊 Driver form/stats updated",
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

    def __init__(self, cfg, poll_interval: int = 3600, db_session_factory=None):
        self.cfg = cfg
        self.poll_interval = max(60, poll_interval)  # Minimum 60 seconds
        self.db_session_factory = db_session_factory
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

    def _get_setting(self, key: str) -> Optional[str]:
        """Fetch a setting from the database."""
        if not self.db_session_factory:
            return None
        from .models_db import Setting
        try:
            with self.db_session_factory() as db:
                setting = db.query(Setting).filter(Setting.key == key).first()
                return setting.value if setting else None
        except Exception as e:
            logger.warning("[PredictionManager] Failed to fetch setting %s: %s", key, e)
            return None

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
            
            # Daily updates for non-next rounds. If interval is 3600s, 24 cycles = 1 day.
            is_next = (round_i == next_r)
            if not is_next:
                if cycle_count % 24 != 0:
                    with self._lock:
                        has_data = str(round_i) in self._latest_results["rounds"]
                    if has_data:
                        continue  # skip to save time

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
        )

        if not results:
            logger.info("[PredictionManager] No results generated for %s", event_title)
            return

        # Fetch actual results to determine if round is frozen and calculate deltas
        actual_results = {}
        try:
            for s in sessions:
                if s == "race":
                    res = jc.get_race_results(str(season_i), str(round_i))
                elif s == "qualifying":
                    res = jc.get_qualifying_results(str(season_i), str(round_i))
                elif s == "sprint":
                    res = jc.get_sprint_results(str(season_i), str(round_i))
                else:
                    res = []

                if res:
                    actual_results[s] = res
        except Exception as e:
            logger.warning("Failed to fetch actual results for %s: %s", event_title, e)

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
        webhook_updates = {}  # sess_key -> (diff, predictions, weather)

        for sess, data in results["sessions"].items():
            ranked_df = data["ranked"]
            ranked_list = ranked_df.to_dict(orient="records")

            # Map actual results
            sess_actuals = actual_results.get(sess, [])
            actual_pos_map = {}
            for row in sess_actuals:
                driver_id = row.get("Driver", {}).get("driverId")
                pos = row.get("position")
                if driver_id and pos:
                    try:
                        actual_pos_map[driver_id] = int(pos)
                    except ValueError:
                        pass

            is_frozen = len(actual_pos_map) > 0

            for row in ranked_list:
                d_id = row.get("driverId")
                if is_frozen and d_id in actual_pos_map:
                    row["actual_position"] = actual_pos_map[d_id]
                row["frozen"] = is_frozen

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

            # --- Diff Detection (Immediate, shared by SSE and webhooks) ---
            diff = None
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

            # --- Webhook Update (Immediate) ---
            # Webhooks are excluded for qualifying sessions.
            # Send on every detected change, matching the SSE diff path.
            if diff and sess not in ("qualifying", "sprint_qualifying"):
                webhook_updates[sess] = (diff, ranked_list, weather)

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

        # Send consolidated Discord webhook if any changes occurred
        if webhook_updates and hasattr(self, '_send_discord_webhook'):
            self._send_discord_webhook(event_title, webhook_updates)

        for diff in all_diffs:
            self._broadcast({
                "type": "diff",
                "data": diff.to_dict(),
                "timestamp": now,
            })
            
        self._broadcast({
            "type": "prediction_round",
            "data": output,
            "timestamp": now,
        })

    def _send_discord_webhook(self, event_title: str, session_updates: dict[str, tuple[PredictionDiff, list[dict[str, Any]], dict[str, Any]]]) -> None:
        """Send a consolidated notification to Discord for all changed sessions in a round."""
        webhook_url = self._get_setting("discord_webhook_url")
        if not webhook_url:
            return

        try:
            embeds = []
            for sess_name, (diff, predictions, weather) in session_updates.items():
                # Format session title (e.g. race -> Race, sprint_qualifying -> Sprint Qualifying)
                display_sess = sess_name.replace("_", " ").title()

                # Build weather description
                weather_str = ""
                if weather and weather.get("temp_mean") is not None:
                    t = weather.get("temp_mean")
                    r = weather.get("rain_sum", 0)
                    w = weather.get("wind_mean", 0)
                    weather_str = f"\n🌡️ **{t:.0f}°C** | 💧 **{r:.1f}mm** | 🌬️ **{w:.0f}km/h**"

                embed = {
                    "author": {"name": "F1 Outcome Predictor"},
                    "title": f"🏁 {event_title} ({display_sess})",
                    "description": f"**Detected changes:** {', '.join(diff.changed_variables)}.{weather_str}",
                    "color": 0xe10600,  # F1 Red
                    "timestamp": diff.timestamp,
                    "fields": [],
                    "footer": {"text": f"Model v{self.cfg.app.model_version}"}
                }

                # Top Movers highlight
                top_gains = sorted([m for m in diff.movements if m.direction > 0],
                                  key=lambda m: m.direction, reverse=True)[:3]
                if top_gains:
                    movers_text = "\n".join([f"⬆️ **{m.code}** (+{m.direction} positions)" for m in top_gains])
                    embed["fields"].append({
                        "name": "🚀 Movers & Shakers",
                        "value": movers_text,
                        "inline": False
                    })

                # Create position maps for movement indicators
                movements = {m.driver_id: m for m in diff.movements}

                # Build the grid display in two columns
                sorted_preds = sorted(predictions, key=lambda x: int(x.get("predicted_position", 99) or 99))

                def format_grid_lines(preds):
                    lines = []
                    for p in preds:
                        pos_val = p.get("predicted_position")
                        pos_str = f"{pos_val:02}" if pos_val is not None else "??"
                        code = p.get("code", "???")
                        d_id = p.get("driverId")

                        m = movements.get(d_id)
                        if m:
                            icon = "⬆️" if m.direction > 0 else "⬇️"
                            diff_val = f"{icon}{abs(m.direction)}"
                        else:
                            diff_val = "⏺️"

                        lines.append(f"`P{pos_str}` **{code}** {diff_val}")
                    return "\n".join(lines) or "No data"

                embed["fields"].append({
                    "name": "Top 10",
                    "value": format_grid_lines(sorted_preds[:10]),
                    "inline": True
                })
                embed["fields"].append({
                    "name": "Bottom 10",
                    "value": format_grid_lines(sorted_preds[10:]),
                    "inline": True
                })

                embeds.append(embed)

            if not embeds:
                return

            # Send up to 10 embeds in one message
            payload = {"embeds": embeds[:10]}
            httpx.post(webhook_url, json=payload, timeout=5.0)
            logger.info("[PredictionManager] Consolidated Discord notification sent for %d sessions", len(session_updates))

        except Exception as e:
            logger.warning("[PredictionManager] Failed to send Discord webhook: %s", e)
