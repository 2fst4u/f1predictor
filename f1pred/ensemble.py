from __future__ import annotations
from dataclasses import dataclass
from typing import Dict

import numpy as np
import pandas as pd

from .util import get_logger

logger = get_logger(__name__)


@dataclass
class EnsembleConfig:
    """Weights for combining individual pace components.

    All weights are relative and are re-normalised to sum to 1.0 at combine-time.
    """

    w_elo: float = 0.25
    w_bt: float = 0.25
    w_mixed: float = 0.25
    w_gbm: float = 0.25
    # Guard rail for std-dev normalisation
    min_std: float = 1e-6


class EloModel:
    """Lightweight Elo model operating purely on historical race results.

    - Ratings are stored per driverId (string).
    - Everyone starts from the same base rating so there is no baked-in prior.
    - Only relative finishing order within each race matters.
    """

    def __init__(self, base_rating: float = 1500.0, k: float = 20.0) -> None:
        self.base_rating = float(base_rating)
        self.k = float(k)
        self.ratings_: Dict[str, float] = {}

    def _get_rating(self, driver_id: str) -> float:
        return self.ratings_.get(driver_id, self.base_rating)

    def fit(self, hist: pd.DataFrame) -> "EloModel":
        if hist is None or hist.empty:
            logger.info("[ensemble.elo] No history; using flat base ratings")
            return self

        races = hist[(hist.get("session") == "race")].dropna(subset=["driverId", "position", "date"]).copy()
        if races.empty:
            logger.info("[ensemble.elo] No race rows; skipping Elo fit")
            return self

        races = races.sort_values("date")
        grouped = races.groupby(["season", "round"])

        updates = 0
        for (_, _), g in grouped:
            g = g.sort_values("position")
            ids = g["driverId"].astype(str).tolist()
            # Pairwise comparisons within the event
            for i in range(len(ids)):
                for j in range(i + 1, len(ids)):
                    a = ids[i]
                    b = ids[j]
                    ra = self._get_rating(a)
                    rb = self._get_rating(b)
                    ea = 1.0 / (1.0 + 10 ** ((rb - ra) / 400.0))
                    eb = 1.0 - ea
                    sa, sb = 1.0, 0.0  # a finished ahead of b
                    self.ratings_[a] = ra + self.k * (sa - ea)
                    self.ratings_[b] = rb + self.k * (sb - eb)
                    updates += 1

        logger.info("[ensemble.elo] Fitted ratings for %d drivers (updates=%d)", len(self.ratings_), updates)
        return self

    def predict(self, X: pd.DataFrame) -> np.ndarray:
        if X is None or X.empty:
            return np.zeros(0, dtype=float)

        ids = X.get("driverId")
        if ids is None:
            logger.info("[ensemble.elo] X missing driverId; returning zeros")
            return np.zeros(len(X), dtype=float)

        scores = [self._get_rating(str(d)) for d in ids]
        arr = np.asarray(scores, dtype=float)
        if arr.size == 0:
            return arr

        # Convert "higher rating is better" to a pace-like index (lower is better)
        mu = float(arr.mean())
        sd = float(arr.std())
        if not np.isfinite(sd) or sd < 1e-6:
            sd = 1.0
        z = -(arr - mu) / sd
        return z


class BradleyTerryModel:
    """Very lightweight Bradley–Terry style strength estimate.

    This is intentionally simple and fast:
      - Uses historical race finishing positions only.
      - Computes mean finishing position per driver as a strength proxy.
      - No priors or hand-tuned tiers; everything comes from results.
    """

    def __init__(self) -> None:
        self.strength_: Dict[str, float] = {}

    def fit(self, hist: pd.DataFrame) -> "BradleyTerryModel":
        if hist is None or hist.empty:
            logger.info("[ensemble.bt] No history; using flat strength")
            return self

        races = hist[(hist.get("session") == "race")].dropna(subset=["driverId", "position"]).copy()
        if races.empty:
            logger.info("[ensemble.bt] No race rows; skipping BT fit")
            return self

        grp = races.groupby("driverId")["position"].mean()
        max_pos = float(grp.max() or 1.0)
        for d, avg_pos in grp.items():
            # smaller average position => higher strength in [0, 1]
            self.strength_[str(d)] = (max_pos - float(avg_pos)) / max_pos

        logger.info("[ensemble.bt] Inferred strengths for %d drivers", len(self.strength_))
        return self

    def predict(self, X: pd.DataFrame) -> np.ndarray:
        if X is None or X.empty:
            return np.zeros(0, dtype=float)
        ids = X.get("driverId")
        if ids is None:
            logger.info("[ensemble.bt] X missing driverId; returning zeros")
            return np.zeros(len(X), dtype=float)

        vals = [self.strength_.get(str(d), 0.0) for d in ids]
        arr = np.asarray(vals, dtype=float)
        if arr.size == 0:
            return arr

        mu = float(arr.mean())
        sd = float(arr.std())
        if not np.isfinite(sd) or sd < 1e-6:
            sd = 1.0
        # Higher strength => better => make into lower-is-better pace index
        z = -(arr - mu) / sd
        return z


class MixedEffectsLikeModel:
    """Approximate driver+constructor decomposition.

    We model a simple additive structure on race performance:

        perf ~= mu + team_effect[constructorId] + driver_effect[driverId]

    where perf = -position (so higher is better). This is *not* a full
    mixed-effects fit, but a fast heuristic that still learns everything
    from results with no hardcoded tiers.
    """

    def __init__(self) -> None:
        self.driver_effect_: Dict[str, float] = {}
        self.team_effect_: Dict[str, float] = {}

    def fit(self, hist: pd.DataFrame) -> "MixedEffectsLikeModel":
        if hist is None or hist.empty:
            logger.info("[ensemble.mixed] No history; using flat effects")
            return self

        races = hist[(hist.get("session") == "race")].dropna(
            subset=["driverId", "constructorId", "position"]
        ).copy()
        if races.empty:
            logger.info("[ensemble.mixed] No race rows; skipping mixed fit")
            return self

        races["perf"] = -races["position"].astype(float)
        mu = float(races["perf"].mean())

        team_mu = races.groupby("constructorId")["perf"].mean()
        team_eff = team_mu - mu
        races = races.join(team_eff.rename("team_eff"), on="constructorId")
        races["team_eff"] = races["team_eff"].fillna(0.0)

        races["driver_resid"] = races["perf"] - mu - races["team_eff"]
        drv_mu = races.groupby("driverId")["driver_resid"].mean()

        self.team_effect_ = {str(k): float(v) for k, v in team_eff.items()}
        self.driver_effect_ = {str(k): float(v) for k, v in drv_mu.items()}

        logger.info(
            "[ensemble.mixed] Fitted %d driver effects, %d team effects",
            len(self.driver_effect_), len(self.team_effect_),
        )
        return self

    def predict(self, X: pd.DataFrame) -> np.ndarray:
        if X is None or X.empty:
            return np.zeros(0, dtype=float)

        vals = []
        for _, row in X.iterrows():
            d = str(row.get("driverId"))
            t = str(row.get("constructorId"))
            de = self.driver_effect_.get(d, 0.0)
            te = self.team_effect_.get(t, 0.0)
            vals.append(de + te)

        arr = np.asarray(vals, dtype=float)
        if arr.size == 0:
            return arr

        mu = float(arr.mean())
        sd = float(arr.std())
        if not np.isfinite(sd) or sd < 1e-6:
            sd = 1.0
        # Higher effect => better => convert to lower-is-better pace index
        z = -(arr - mu) / sd
        return z


def _safe_component(arr: np.ndarray | None, n: int, name: str) -> np.ndarray:
    if arr is None:
        logger.info("[ensemble.combine] %s unavailable; using zeros", name)
        return np.zeros(n, dtype=float)
    if len(arr) != n:
        logger.info("[ensemble.combine] %s length %d != %d; using zeros", name, len(arr), n)
        return np.zeros(n, dtype=float)
    return arr.astype(float)


def combine_pace(
    gbm_pace: np.ndarray,
    elo_pace: np.ndarray | None,
    bt_pace: np.ndarray | None,
    mixed_pace: np.ndarray | None,
    cfg: EnsembleConfig | None = None,
) -> np.ndarray:
    """Combine multiple pace components into a single pace index.

    - All inputs are assumed to be on a "lower is better" scale.
    - Components are z-scored implicitly via their contribution.
    - If any component is missing or has wrong length, it is safely replaced
      with zeros and a debug log is emitted.
    """

    if gbm_pace is None:
        logger.info("[ensemble.combine] gbm_pace is None; returning empty array")
        return np.zeros(0, dtype=float)

    n = len(gbm_pace)
    if n == 0:
        return np.zeros(0, dtype=float)

    cfg = cfg or EnsembleConfig()

    g = _safe_component(gbm_pace, n, "gbm")
    e = _safe_component(elo_pace, n, "elo") if elo_pace is not None else np.zeros(n, dtype=float)
    b = _safe_component(bt_pace, n, "bt") if bt_pace is not None else np.zeros(n, dtype=float)
    m = _safe_component(mixed_pace, n, "mixed") if mixed_pace is not None else np.zeros(n, dtype=float)

    wsum = cfg.w_elo + cfg.w_bt + cfg.w_mixed + cfg.w_gbm
    if wsum <= 0:
        logger.info("[ensemble.combine] Non-positive weight sum; falling back to gbm only")
        w_gbm = 1.0
        w_elo = w_bt = w_mixed = 0.0
    else:
        w_gbm = cfg.w_gbm / wsum
        w_elo = cfg.w_elo / wsum
        w_bt = cfg.w_bt / wsum
        w_mixed = cfg.w_mixed / wsum

    combined = w_gbm * g + w_elo * e + w_bt * b + w_mixed * m

    mu = float(combined.mean())
    sd = float(combined.std())
    if not np.isfinite(sd) or sd < cfg.min_std:
        sd = 1.0
    z = (combined - mu) / sd

    # Add deterministic tiny jitter to break exact ties in a reproducible way
    jitter = (np.arange(n, dtype=float) % 997) / 1e6
    out = z + jitter

    logger.info(
        "[ensemble.combine] Combined pace: std=%.6f, range=%.6f", float(out.std()), float(np.ptp(out))
    )
    return out
