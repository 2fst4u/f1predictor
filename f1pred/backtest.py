from __future__ import annotations
from typing import List, Dict, Any, Optional
from datetime import datetime, timezone
from pathlib import Path
import json

from .util import get_logger
from .data.jolpica import JolpicaClient
from .predict import run_predictions_for_event
from .metrics import compute_event_metrics

logger = get_logger(__name__)

# Per-metric optimisation direction so before/after comparisons know which way
# is an improvement.  "higher" => bigger is better; "lower" => smaller is better.
METRIC_DIRECTION: Dict[str, str] = {
    "spearman": "higher",
    "kendall": "higher",
    "accuracy_top3": "higher",
    "brier_pairwise": "lower",
    "crps": "lower",
}


def _auto_backtest_seasons(jc: JolpicaClient) -> List[int]:
    try:
        js = jc.get_season_schedule("current")
        current_year = int(js[0]["season"])
        years = list(range(current_year - 5, current_year))
        return years
    except Exception:
        return [2020, 2021, 2022, 2023]


def _is_nan(x: Any) -> bool:
    return isinstance(x, float) and x != x


def summarize_metrics(metrics_rows: List[Dict[str, Any]]) -> Dict[str, Any]:
    """Aggregate per-session metric rows into mean/median summary statistics.

    NaNs and missing keys are skipped per metric.  Results are grouped overall
    and broken out per session type (race/qualifying/sprint/...) so weather- and
    grid-sensitive sessions can be inspected separately.
    """
    import numpy as np

    def _agg(rows: List[Dict[str, Any]]) -> Dict[str, Any]:
        per_metric: Dict[str, Any] = {}
        for key in METRIC_DIRECTION:
            vals = [r[key] for r in rows
                    if key in r and r[key] is not None and not _is_nan(r[key])]
            if vals:
                per_metric[key] = {
                    "mean": float(np.mean(vals)),
                    "median": float(np.median(vals)),
                    "n": len(vals),
                }
        return per_metric

    by_session: Dict[str, List[Dict[str, Any]]] = {}
    for r in metrics_rows:
        by_session.setdefault(str(r.get("event", "unknown")), []).append(r)

    return {
        "n_sessions": len(metrics_rows),
        "generated_at": datetime.now(timezone.utc).isoformat(),
        "overall": _agg(metrics_rows),
        "by_session": {sess: _agg(rows) for sess, rows in by_session.items()},
    }


def compare_summaries(baseline: Dict[str, Any], candidate: Dict[str, Any]) -> Dict[str, Any]:
    """Diff two summaries (candidate - baseline) with improvement direction.

    Returns a per-metric dict of baseline/candidate means, the delta, and whether
    the candidate improved on that metric given its optimisation direction.
    """
    b = baseline.get("overall", {})
    c = candidate.get("overall", {})
    out: Dict[str, Any] = {}
    for key, direction in METRIC_DIRECTION.items():
        if key in b and key in c:
            bm = b[key]["mean"]
            cm = c[key]["mean"]
            delta = cm - bm
            improved = delta > 0 if direction == "higher" else delta < 0
            out[key] = {
                "baseline": bm,
                "candidate": cm,
                "delta": delta,
                "direction": direction,
                "improved": bool(improved),
            }
    return out


def _format_comparison(cmp: Dict[str, Any]) -> str:
    lines = ["", "Backtest comparison (candidate vs baseline):",
             f"  {'metric':<16}{'baseline':>12}{'candidate':>12}{'delta':>12}  verdict"]
    for key, d in cmp.items():
        verdict = "better" if d["improved"] else "worse/flat"
        arrow = "↑" if d["direction"] == "higher" else "↓"
        lines.append(
            f"  {key + ' ' + arrow:<16}{d['baseline']:>12.4f}{d['candidate']:>12.4f}"
            f"{d['delta']:>+12.4f}  {verdict}"
        )
    return "\n".join(lines)


def _persist(cfg, metrics_rows: List[Dict[str, Any]], summary: Dict[str, Any],
             label: Optional[str]) -> Optional[str]:
    """Write per-event CSV + summary JSON under <cache_dir>/backtests. Best-effort."""
    try:
        import pandas as pd
        out_dir = Path(cfg.paths.cache_dir) / "backtests"
        out_dir.mkdir(parents=True, exist_ok=True)
        tag = label or datetime.now(timezone.utc).strftime("%Y%m%dT%H%M%SZ")
        if metrics_rows:
            pd.DataFrame(metrics_rows).to_csv(out_dir / f"backtest_{tag}.csv", index=False)
        summary_path = out_dir / f"backtest_{tag}.summary.json"
        with open(summary_path, "w") as f:
            json.dump(summary, f, indent=2)
        logger.info("[backtest] Wrote summary to %s", summary_path)
        return str(summary_path)
    except Exception as e:
        logger.warning("[backtest] Could not persist results: %s", e)
        return None


def run_backtests(cfg, label: Optional[str] = None,
                  baseline_summary: Optional[Dict[str, Any]] = None) -> Dict[str, Any]:
    """Run rolling backtests, aggregate accuracy metrics, and persist a summary.

    Args:
        cfg: Application config.
        label: Optional tag for the output files (e.g. "before" / "after").
        baseline_summary: If given, a comparison vs this summary is logged and
            returned under the ``"comparison"`` key — the quick before/after view.

    Returns the aggregated summary dict (also written to
    ``<cache_dir>/backtests/``).
    """
    jc = JolpicaClient(cfg.data_sources.jolpica.base_url, cfg.data_sources.jolpica.timeout_seconds,
                       cfg.data_sources.jolpica.rate_limit_sleep)
    seasons = cfg.backtesting.seasons
    if seasons == "auto":
        season_list = _auto_backtest_seasons(jc)
    elif seasons == "all":
        season_list = list(range(1950, datetime.now(timezone.utc).year))
    elif isinstance(seasons, list):
        season_list = [int(x) for x in seasons]
    else:
        season_list = _auto_backtest_seasons(jc)

    metrics_rows: List[Dict[str, Any]] = []

    for season in season_list:
        races = jc.get_season_schedule(str(season))
        for r in races:
            rnd = r["round"]
            logger.info(f"Backtest: {season} R{rnd} {r['raceName']}")
            try:
                res = run_predictions_for_event(cfg, season=str(season), rnd=str(rnd),
                                                sessions=cfg.modelling.targets.session_types,
                                                return_results=True,
                                                use_actuals=False)
                if not res:
                    continue
                for sess, sdata in res["sessions"].items():
                    ranked = sdata["ranked"]
                    prob_matrix = sdata["prob_matrix"]
                    pairwise = sdata["pairwise"]
                    mrow = compute_event_metrics(ranked, prob_matrix, pairwise, session=sess,
                                                 season=res["season"], rnd=res["round"])
                    metrics_rows.append(mrow)

            except Exception as e:
                logger.warning(f"Backtest prediction failed for {season} R{rnd}: {e}")
                continue

    logger.info(f"Backtesting complete. Processed {len(metrics_rows)} session predictions.")

    summary = summarize_metrics(metrics_rows)
    _persist(cfg, metrics_rows, summary, label)

    # Log a readable overall table.
    overall = summary.get("overall", {})
    if overall:
        logger.info("[backtest] Summary over %d sessions:", summary["n_sessions"])
        for key in METRIC_DIRECTION:
            if key in overall:
                logger.info("  %-16s mean=%.4f median=%.4f (n=%d)",
                            key, overall[key]["mean"], overall[key]["median"], overall[key]["n"])

    if baseline_summary is not None:
        cmp = compare_summaries(baseline_summary, summary)
        summary["comparison"] = cmp
        logger.info("%s", _format_comparison(cmp))

    return summary
