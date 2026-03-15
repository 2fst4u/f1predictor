from __future__ import annotations
from typing import List, Optional
import os

import json
import threading
import queue
import math
from fastapi import FastAPI, Request, Query, Path, HTTPException
from fastapi.responses import HTMLResponse, StreamingResponse
from fastapi.templating import Jinja2Templates

from .config import AppConfig
from .util import get_logger, init_caches, ensure_dirs, __version__
from .predict import run_predictions_for_event, resolve_event
from .data.jolpica import JolpicaClient
from .data.fastf1_backend import init_fastf1

logger = get_logger(__name__)

# Global app and config
app = FastAPI(title="F1 Prediction Web UI")
_config: AppConfig = None

@app.middleware("http")
async def add_security_headers(request: Request, call_next):
    response = await call_next(request)
    # 🛡️ Sentinel: Defense in depth - Add security headers to prevent common web vulnerabilities
    response.headers["X-Content-Type-Options"] = "nosniff"
    response.headers["X-Frame-Options"] = "DENY"
    response.headers["X-XSS-Protection"] = "1; mode=block"
    response.headers["Strict-Transport-Security"] = "max-age=31536000; includeSubDomains"

    # CSP: Allow self, CDNs for Tailwind/Alpine/FontAwesome, and inline scripts/styles for Alpine/Tailwind.
    csp = (
        "default-src 'self'; "
        "script-src 'self' 'unsafe-inline' 'unsafe-eval' https://cdn.tailwindcss.com https://cdn.jsdelivr.net; "
        "style-src 'self' 'unsafe-inline' https://cdnjs.cloudflare.com https://fonts.googleapis.com; "
        "font-src 'self' https://cdnjs.cloudflare.com https://fonts.gstatic.com; "
        "connect-src 'self';"
    )
    response.headers["Content-Security-Policy"] = csp
    return response

# Templates
templates = Jinja2Templates(directory=os.path.join(os.path.dirname(__file__), "templates"))

def init_web(cfg: AppConfig):
    global _config
    _config = cfg

    # Initialize directories (include fastf1_cache so init_fastf1 finds it)
    ensure_dirs(cfg.paths.cache_dir, cfg.paths.fastf1_cache)

    # Initialize FastF1 before requests_cache to avoid MRO conflicts
    try:
        if cfg.data_sources.fastf1.enabled:
            init_fastf1(cfg.paths.fastf1_cache)
    except Exception as e:
        logger.warning(f"FastF1 initialization failed: {e}")

    # Initialize HTTP cache
    init_caches(cfg)

@app.get("/", response_class=HTMLResponse)
async def index(request: Request):
    return templates.TemplateResponse(request, "index.html")

@app.get("/api/config")
async def get_web_config():
    if not _config:
        return {"error": "Config not initialized"}

    # Get next round info for pre-selection
    jc = JolpicaClient(_config.data_sources.jolpica.base_url)
    try:
        next_s, next_r, _ = resolve_event(jc, "current", "next")
    except Exception:
        next_s, next_r = None, None

    return {
        "model_version": _config.app.model_version,
        "app_version": __version__,
        "default_sessions": _config.modelling.targets.session_types,
        "next_event": {
            "season": str(next_s) if next_s is not None else None,
            "round": str(next_r) if next_r is not None else None,
        }
    }

@app.get("/api/seasons")
async def get_seasons():
    jc = JolpicaClient(_config.data_sources.jolpica.base_url)
    try:
        seasons = jc.get_seasons()
        # Return in descending order for better UX (newest first)
        seasons.sort(key=lambda x: x.get("season", "0"), reverse=True)
        return seasons
    except Exception:
        logger.exception("Failed to get seasons")
        raise HTTPException(status_code=500, detail="Internal server error")

@app.get("/api/schedule/{season}")
async def get_schedule(season: str = Path(..., max_length=10)):
    jc = JolpicaClient(_config.data_sources.jolpica.base_url)
    try:
        races = jc.get_season_schedule(season)
        return {"season": season, "races": races}
    except Exception:
        logger.exception("Failed to get schedule")
        raise HTTPException(status_code=500, detail="Internal server error")

@app.get("/api/event-status/{season}/{round}")
async def get_event_status(
    season: str = Path(..., max_length=10),
    round: str = Path(..., max_length=10)
):
    if not _config:
        raise HTTPException(status_code=500, detail="Application not configured")

    jc = JolpicaClient(
        _config.data_sources.jolpica.base_url,
        _config.data_sources.jolpica.timeout_seconds,
        _config.data_sources.jolpica.rate_limit_sleep,
    )

    try:
        # Resolve actual season/round numbers
        s_i, r_i, race_info = resolve_event(jc, season, round)

        # Standard chronological order
        all_possible = ["sprint_qualifying", "sprint", "qualifying", "race"]
        sessions = []

        # Derive roster once for the round (Optimization: avoid redundant deep scans)
        from .features import build_roster
        roster = build_roster(jc, str(s_i), str(r_i), event_dt=None)

        # Check for each session if it exists in race_info and if results exist
        for s in all_possible:
            # Race and Qualifying always exist in F1
            exists = s in ("race", "qualifying")

            # Sprint sessions only if present in race_info
            if s == "sprint" and "Sprint" in race_info:
                exists = True
            if s == "sprint_qualifying" and "SprintQualifying" in race_info:
                exists = True

            if exists:
                has_results = False

                # 1. Primary check: Use prediction engine results (Jolpica + FastF1)
                try:
                    from .predict import _get_actual_positions_for_session
                    if roster is not None and not roster.empty:
                        acts = _get_actual_positions_for_session(jc, s_i, r_i, s, roster)
                        has_results = acts is not None and not acts.isna().all()
                except Exception:
                    pass

                # 2. Secondary check: Fresh Jolpica query (bypass cache to
                #    avoid stale empty responses from before results were posted)
                if not has_results:
                    try:
                        import requests_cache as rc
                        with rc.disabled():
                            if s == "race":
                                has_results = bool(jc.get_race_results(str(s_i), str(r_i)))
                            elif s == "qualifying":
                                has_results = bool(jc.get_qualifying_results(str(s_i), str(r_i)))
                            elif s == "sprint":
                                has_results = bool(jc.get_sprint_results(str(s_i), str(r_i)))
                            elif s == "sprint_qualifying":
                                # No Jolpica endpoint for sprint qualifying; rely on
                                # the primary check above.
                                pass
                    except Exception:
                        pass

                sessions.append({
                    "id": s,
                    "name": s.replace("_", " ").title(),
                    "has_results": has_results
                })

        return {
            "season": s_i,
            "round": r_i,
            "raceName": race_info.get("raceName"),
            "sessions": sessions
        }
    except Exception:
        logger.exception("Failed to get event status")
        raise HTTPException(status_code=500, detail="Internal server error")

@app.get("/api/predict")
async def get_predictions(
    season: Optional[str] = Query(None, max_length=10),
    round: str = Query("next", max_length=10),
    sessions: List[str] = Query(None, max_length=20)
):
    if not _config:
         raise HTTPException(status_code=500, detail="Application not configured")

    try:
        # Use default sessions if not provided
        target_sessions = sessions or _config.modelling.targets.session_types

        # run_predictions_for_event is blocking, run in threadpool
        # But for now, just call it. FastAPI handles sync routes in threads.
        results = run_predictions_for_event(
            _config,
            season=season,
            rnd=round,
            sessions=target_sessions,
            return_results=True
        )

        if not results:
            return {"error": "No results generated"}

        # Convert results to JSON-serializable format
        output = {
            "season": results["season"],
            "round": results["round"],
            "sessions": {}
        }

        for sess, data in results["sessions"].items():
            ranked_df = data["ranked"]
            # Convert DataFrame to list of dicts
            ranked_list = ranked_df.to_dict(orient="records")
            # Clean up NaNs for JSON
            import math
            for row in ranked_list:
                for k, v in row.items():
                    if isinstance(v, float) and (math.isnan(v) or math.isinf(v)):
                        row[k] = None
                    elif hasattr(v, "isoformat"):
                        row[k] = v.isoformat()

            output["sessions"][sess] = {
                "predictions": ranked_list,
                "weather": data.get("meta", {}).get("weather", {})
            }

        return output

    except Exception:
        logger.exception("Prediction failed")
        raise HTTPException(status_code=500, detail="Internal server error")

@app.get("/api/predict/stream")
async def get_predictions_stream(
    season: Optional[str] = Query(None, max_length=10),
    round: str = Query("next", max_length=10),
    sessions: List[str] = Query(None, max_length=20)
):
    if not _config:
         raise HTTPException(status_code=500, detail="Application not configured")

    def run_and_collect(q: queue.Queue):
        try:
            target_sessions = sessions or _config.modelling.targets.session_types

            def progress_cb(msg: str):
                q.put({"type": "log", "message": msg})

            results = run_predictions_for_event(
                _config,
                season=season,
                rnd=round,
                sessions=target_sessions,
                return_results=True,
                progress_callback=progress_cb
            )

            if not results:
                q.put({"type": "error", "message": "No results generated"})
                return

            # Convert results to JSON-serializable format (similar to /api/predict)
            output = {
                "season": results["season"],
                "round": results["round"],
                "sessions": {}
            }

            for sess, data in results["sessions"].items():
                ranked_df = data["ranked"]
                ranked_list = ranked_df.to_dict(orient="records")
                for row in ranked_list:
                    for k, v in row.items():
                        if isinstance(v, float) and (math.isnan(v) or math.isinf(v)):
                            row[k] = None
                        elif hasattr(v, "isoformat"):
                            row[k] = v.isoformat()

                output["sessions"][sess] = {
                    "predictions": ranked_list,
                    "weather": data.get("meta", {}).get("weather", {})
                }

            q.put({"type": "results", "data": output})
        except Exception:
            logger.exception("Streaming prediction failed")
            q.put({"type": "error", "message": "Internal server error"})
        finally:
            q.put(None) # Sentinel to close stream

    def event_generator():
        q = queue.Queue()
        thread = threading.Thread(target=run_and_collect, args=(q,))
        thread.start()

        while True:
            item = q.get()
            if item is None:
                break

            yield f"data: {json.dumps(item)}\n\n"

    return StreamingResponse(event_generator(), media_type="text/event-stream")
