from __future__ import annotations
from dataclasses import dataclass
from typing import Dict, Optional, TYPE_CHECKING

from datetime import timezone

from .util import get_logger

if TYPE_CHECKING:
    import numpy as np
    import pandas as pd

logger = get_logger(__name__)

_QUALI_SESSIONS = frozenset({"qualifying", "sprint_qualifying"})
_RACE_SESSIONS = frozenset({"race", "sprint"})


@dataclass
class EnsembleConfig:
    """Weights for combining individual pace components.

    Separate weight sets are held for race-like sessions and qualifying-like
    sessions so that models trained purely on race results (Elo, BT, Mixed)
    can be down-weighted when predicting qualifying — where their race-derived
    signal is less informative — without degrading race predictions.

    All weights within each set are relative and are re-normalised to sum to
    1.0 at combine-time.  Defaults match config.yaml so that uncalibrated
    runs behave consistently.
    """

    # Race / sprint weights
    w_gbm: float = 0.4
    w_elo: float = 0.2
    w_bt: float = 0.2
    w_mixed: float = 0.2

    # Qualifying / sprint_qualifying weights
    # By default the GBM (which switches to qualifying_form_index) carries
    # substantially more weight; the race-only statistical models are kept at
    # a small but non-zero contribution so they still provide a regularising
    # signal from long-run driver/team pace.
    w_gbm_quali: float = 0.7
    w_elo_quali: float = 0.1
    w_bt_quali: float = 0.1
    w_mixed_quali: float = 0.1

    # Guard rail for std-dev normalisation (matches config.yaml modelling.ensemble.min_std)
    min_std: float = 0.05


class EloModel:
    """Elo rating model with separate race and qualifying rating tracks.

    Race ratings are updated from race finishing order (as before).
    Qualifying ratings are updated from qualifying/sprint_qualifying position
    order so they reflect pure one-lap pace rather than race-day execution.

    ``fit()`` processes all available sessions in a single pass.
    ``predict()`` selects the appropriate rating track via ``session_type``.
    """

    def __init__(self, base_rating: float = 1500.0, k: float = 20.0) -> None:
        self.base_rating = float(base_rating)
        self.k = float(k)
        self.race_ratings_: Dict[str, float] = {}
        self.quali_ratings_: Dict[str, float] = {}

    def _get_race_rating(self, driver_id: str) -> float:
        return self.race_ratings_.get(driver_id, self.base_rating)

    def _get_quali_rating(self, driver_id: str) -> float:
        return self.quali_ratings_.get(driver_id, self.base_rating)

    def _update_ratings(
        self,
        ratings: Dict[str, float],
        ordered_ids: list,
    ) -> None:
        """Apply pairwise Elo updates in-place for a single session."""
        for i in range(len(ordered_ids)):
            for j in range(i + 1, len(ordered_ids)):
                a = ordered_ids[i]
                b = ordered_ids[j]
                ra = ratings.get(a, self.base_rating)
                rb = ratings.get(b, self.base_rating)
                ea = 1.0 / (1.0 + 10 ** ((rb - ra) / 400.0))
                eb = 1.0 - ea
                ratings[a] = ra + self.k * (1.0 - ea)
                ratings[b] = rb + self.k * (0.0 - eb)

    def fit(self, hist: "pd.DataFrame") -> "EloModel":
        """Fit both race and qualifying Elo tracks from historical results."""
        if hist is None or hist.empty:
            logger.info("[ensemble.elo] No history; using flat base ratings")
            return self

        race_rows = (
            hist[hist["session"].isin(_RACE_SESSIONS)]
            .dropna(subset=["driverId", "position", "date"])
            .copy()
        )
        # Guard: qpos column may not exist in older/mock history DataFrames
        if "qpos" in hist.columns:
            quali_rows = (
                hist[hist["session"].isin(_QUALI_SESSIONS)]
                .dropna(subset=["driverId", "qpos", "date"])
                .copy()
            )
        else:
            import pandas as _pd
            quali_rows = _pd.DataFrame()

        # --- Race Elo ---
        race_updates = 0
        if not race_rows.empty:
            for (_, _), g in race_rows.sort_values("date").groupby(
                ["season", "round"], sort=False
            ):
                ids = g.sort_values("position")["driverId"].astype(str).tolist()
                self._update_ratings(self.race_ratings_, ids)
                race_updates += len(ids) * (len(ids) - 1) // 2

        # --- Qualifying Elo ---
        quali_updates = 0
        if not quali_rows.empty:
            for (_, _), g in quali_rows.sort_values("date").groupby(
                ["season", "round"], sort=False
            ):
                ids = g.sort_values("qpos")["driverId"].astype(str).tolist()
                self._update_ratings(self.quali_ratings_, ids)
                quali_updates += len(ids) * (len(ids) - 1) // 2

        logger.info(
            "[ensemble.elo] Race ratings: %d drivers (%d updates); "
            "Quali ratings: %d drivers (%d updates)",
            len(self.race_ratings_),
            race_updates,
            len(self.quali_ratings_),
            quali_updates,
        )
        return self

    # ------------------------------------------------------------------
    # Backward-compatible aliases (old code accessed model.ratings_)
    # ------------------------------------------------------------------
    @property
    def ratings_(self) -> Dict[str, float]:
        """Alias for race_ratings_ (backward compatibility)."""
        return self.race_ratings_

    @ratings_.setter
    def ratings_(self, value: Dict[str, float]) -> None:
        self.race_ratings_ = value

    def predict(
        self, X: "pd.DataFrame", session_type: str = "race"
    ) -> "np.ndarray":
        """Return a lower-is-better pace z-score from the appropriate rating track."""
        import numpy as np

        if X is None or X.empty:
            return np.zeros(0, dtype=float)

        ids = X.get("driverId")
        if ids is None:
            logger.info("[ensemble.elo] X missing driverId; returning zeros")
            return np.zeros(len(X), dtype=float)

        is_quali = session_type in _QUALI_SESSIONS
        _ = self.quali_ratings_ if is_quali else self.race_ratings_
        get_fn = self._get_quali_rating if is_quali else self._get_race_rating

        scores = [get_fn(str(d)) for d in ids]
        arr = np.asarray(scores, dtype=float)
        if arr.size == 0:
            return arr

        mu = float(arr.mean())
        sd = float(arr.std())
        if not np.isfinite(sd) or sd < 1e-6:
            sd = 1.0
        # Higher rating → better → negate to get lower-is-better
        return -(arr - mu) / sd


class BradleyTerryModel:
    """Session-aware Bradley–Terry style strength estimate.

    For race/sprint sessions the model uses finishing position (as before).
    For qualifying/sprint_qualifying sessions it uses qualifying position
    (``qpos``) so the strength estimate reflects pure one-lap pace.

    A single ``fit()`` call builds both tracks simultaneously.
    ``predict()`` selects the appropriate track via ``session_type``.
    """

    def __init__(self) -> None:
        self.race_strength_: Dict[str, float] = {}
        self.quali_strength_: Dict[str, float] = {}

    @staticmethod
    def _compute_strength(
        rows: "pd.DataFrame",
        pos_col: str,
        ref_date: "pd.Timestamp",
        half_life_days: float,
    ) -> Dict[str, float]:
        """Compute recency-weighted mean-position strength for *rows*."""
        import numpy as np
        import pandas as pd

        if rows.empty:
            return {}

        dates = pd.to_datetime(rows["date"], utc=True)
        diff = ref_date.to_datetime64() - dates.values
        ages = diff.astype("timedelta64[ns]").astype(float) / 86_400_000_000_000.0
        w = np.exp2(-ages / max(1.0, half_life_days))

        driver_ids, group_idx = np.unique(rows["driverId"].values, return_inverse=True)
        n_groups = len(driver_ids)

        pos_vals = rows[pos_col].values.astype(float)
        w_pos = pos_vals * w

        w_sum = np.bincount(group_idx, weights=w, minlength=n_groups)
        w_pos_sum = np.bincount(group_idx, weights=w_pos, minlength=n_groups)

        w_sum_safe = np.maximum(w_sum, 1e-6)
        w_mean = w_pos_sum / w_sum_safe

        max_pos = float(pos_vals.max() or 20.0)
        str_vals = (max_pos - w_mean) / max_pos

        return {str(k): float(v) for k, v in zip(driver_ids, str_vals)}

    def fit(
        self, hist: "pd.DataFrame", half_life_days: float = 365.0
    ) -> "BradleyTerryModel":
        import pandas as pd

        if hist is None or hist.empty:
            logger.info("[ensemble.bt] No history; using flat strength")
            return self

        ref_date = pd.Timestamp(hist["date"].max()).tz_convert(timezone.utc)

        # Race track
        race_rows = (
            hist[hist["session"].isin(_RACE_SESSIONS)]
            .dropna(subset=["driverId", "position"])
            .copy()
        )
        if race_rows.empty:
            logger.info("[ensemble.bt] No race rows; skipping race BT fit")
        else:
            self.race_strength_ = self._compute_strength(
                race_rows, "position", ref_date, half_life_days
            )
            logger.info(
                "[ensemble.bt] Race strengths for %d drivers", len(self.race_strength_)
            )

        # Qualifying track — guard against history DataFrames without qpos
        if "qpos" in hist.columns:
            quali_rows = (
                hist[hist["session"].isin(_QUALI_SESSIONS)]
                .dropna(subset=["driverId", "qpos"])
                .copy()
            )
        else:
            quali_rows = pd.DataFrame()
        if quali_rows.empty:
            logger.info("[ensemble.bt] No qualifying rows; skipping quali BT fit")
        else:
            self.quali_strength_ = self._compute_strength(
                quali_rows, "qpos", ref_date, half_life_days
            )
            logger.info(
                "[ensemble.bt] Quali strengths for %d drivers",
                len(self.quali_strength_),
            )

        return self

    # ------------------------------------------------------------------
    # Backward-compatible alias (old code accessed model.strength_)
    # ------------------------------------------------------------------
    @property
    def strength_(self) -> Dict[str, float]:
        """Alias for race_strength_ (backward compatibility)."""
        return self.race_strength_

    @strength_.setter
    def strength_(self, value: Dict[str, float]) -> None:
        self.race_strength_ = value

    def predict(
        self, X: "pd.DataFrame", session_type: str = "race"
    ) -> "np.ndarray":
        import numpy as np

        if X is None or X.empty:
            return np.zeros(0, dtype=float)
        ids = X.get("driverId")
        if ids is None:
            logger.info("[ensemble.bt] X missing driverId; returning zeros")
            return np.zeros(len(X), dtype=float)

        is_quali = session_type in _QUALI_SESSIONS
        strength = self.quali_strength_ if is_quali else self.race_strength_

        # Fall back to race strength when qualifying track is empty
        if not strength:
            strength = self.race_strength_

        vals = [strength.get(str(d), 0.0) for d in ids]
        arr = np.asarray(vals, dtype=float)
        if arr.size == 0:
            return arr

        mu = float(arr.mean())
        sd = float(arr.std())
        if not np.isfinite(sd) or sd < 1e-6:
            sd = 1.0
        # Higher strength → better → negate to lower-is-better
        return -(arr - mu) / sd


class MixedEffectsLikeModel:
    """Session-aware driver + constructor additive decomposition.

    For race/sprint sessions:
        perf = mu + team_effect[constructor] + driver_effect[driver]
        perf = -position

    For qualifying/sprint_qualifying sessions:
        quali_perf = mu + team_quali_effect[constructor] + driver_quali_effect[driver]
        quali_perf = -qpos

    Both tracks are fitted in a single ``fit()`` call.  ``predict()`` selects
    the appropriate track via ``session_type``.
    """

    def __init__(self) -> None:
        self.race_driver_effect_: Dict[str, float] = {}
        self.race_team_effect_: Dict[str, float] = {}
        self.quali_driver_effect_: Dict[str, float] = {}
        self.quali_team_effect_: Dict[str, float] = {}

    @staticmethod
    def _compute_effects(
        rows: "pd.DataFrame",
        perf_col: str,
        ref_date: "pd.Timestamp",
        half_life_days: float,
    ):
        """Return (driver_effect, team_effect) dicts for *rows*."""
        import numpy as np
        import pandas as pd

        if rows.empty:
            return {}, {}

        dates = pd.to_datetime(rows["date"], utc=True)
        diff = ref_date.to_datetime64() - dates.values
        ages = diff.astype("timedelta64[ns]").astype(float) / 86_400_000_000_000.0
        w = np.exp2(-ages / max(1.0, half_life_days))

        perf_vals = rows[perf_col].values.astype(float)
        mu = float(np.average(perf_vals, weights=w))

        # Team effects
        team_ids, team_idx = np.unique(rows["constructorId"].values, return_inverse=True)
        n_teams = len(team_ids)
        w_perf = perf_vals * w
        team_w_sum = np.bincount(team_idx, weights=w, minlength=n_teams)
        team_w_perf_sum = np.bincount(team_idx, weights=w_perf, minlength=n_teams)
        team_mu = team_w_perf_sum / np.maximum(team_w_sum, 1e-6)
        team_eff_vals = team_mu - mu

        # Driver residuals
        row_team_eff = team_eff_vals[team_idx]
        driver_resid = perf_vals - mu - row_team_eff

        drv_ids, drv_idx = np.unique(rows["driverId"].values, return_inverse=True)
        n_drvs = len(drv_ids)
        w_resid = driver_resid * w
        drv_w_sum = np.bincount(drv_idx, weights=w, minlength=n_drvs)
        drv_w_resid_sum = np.bincount(drv_idx, weights=w_resid, minlength=n_drvs)
        drv_mu = drv_w_resid_sum / np.maximum(drv_w_sum, 1e-6)

        driver_effect = {str(k): float(v) for k, v in zip(drv_ids, drv_mu)}
        team_effect = {str(k): float(v) for k, v in zip(team_ids, team_eff_vals)}
        return driver_effect, team_effect

    def fit(
        self, hist: "pd.DataFrame", half_life_days: float = 365.0
    ) -> "MixedEffectsLikeModel":
        import pandas as pd

        if hist is None or hist.empty:
            logger.info("[ensemble.mixed] No history; using flat effects")
            return self

        ref_date = pd.Timestamp(hist["date"].max()).tz_convert(timezone.utc)

        # Race track
        race_rows = (
            hist[hist["session"].isin(_RACE_SESSIONS)]
            .dropna(subset=["driverId", "constructorId", "position"])
            .copy()
        )
        if race_rows.empty:
            logger.info("[ensemble.mixed] No race rows; skipping race mixed fit")
        else:
            race_rows["perf"] = -race_rows["position"].astype(float)
            self.race_driver_effect_, self.race_team_effect_ = self._compute_effects(
                race_rows, "perf", ref_date, half_life_days
            )
            logger.info(
                "[ensemble.mixed] Race: %d driver effects, %d team effects",
                len(self.race_driver_effect_),
                len(self.race_team_effect_),
            )

        # Qualifying track — guard against history DataFrames without qpos
        if "qpos" in hist.columns:
            quali_rows = (
                hist[hist["session"].isin(_QUALI_SESSIONS)]
                .dropna(subset=["driverId", "constructorId", "qpos"])
                .copy()
            )
        else:
            quali_rows = pd.DataFrame()
        if quali_rows.empty:
            logger.info("[ensemble.mixed] No qualifying rows; skipping quali mixed fit")
        else:
            quali_rows["perf"] = -quali_rows["qpos"].astype(float)
            self.quali_driver_effect_, self.quali_team_effect_ = self._compute_effects(
                quali_rows, "perf", ref_date, half_life_days
            )
            logger.info(
                "[ensemble.mixed] Quali: %d driver effects, %d team effects",
                len(self.quali_driver_effect_),
                len(self.quali_team_effect_),
            )

        return self

    # ------------------------------------------------------------------
    # Backward-compatible aliases (old code accessed model.driver_effect_
    # and model.team_effect_ directly)
    # ------------------------------------------------------------------
    @property
    def driver_effect_(self) -> Dict[str, float]:
        """Alias for race_driver_effect_ (backward compatibility)."""
        return self.race_driver_effect_

    @driver_effect_.setter
    def driver_effect_(self, value: Dict[str, float]) -> None:
        self.race_driver_effect_ = value

    @property
    def team_effect_(self) -> Dict[str, float]:
        """Alias for race_team_effect_ (backward compatibility)."""
        return self.race_team_effect_

    @team_effect_.setter
    def team_effect_(self, value: Dict[str, float]) -> None:
        self.race_team_effect_ = value

    def predict(
        self, X: "pd.DataFrame", session_type: str = "race"
    ) -> "np.ndarray":
        import numpy as np

        if X is None or X.empty:
            return np.zeros(0, dtype=float)

        is_quali = session_type in _QUALI_SESSIONS
        if is_quali and (self.quali_driver_effect_ or self.quali_team_effect_):
            driver_effect = self.quali_driver_effect_
            team_effect = self.quali_team_effect_
        else:
            # Fall back to race track when qualifying track is empty
            driver_effect = self.race_driver_effect_
            team_effect = self.race_team_effect_

        vals = []
        for _, row in X.iterrows():
            d = str(row.get("driverId"))
            t = str(row.get("constructorId"))
            de = driver_effect.get(d, 0.0)
            te = team_effect.get(t, 0.0)
            vals.append(de + te)

        arr = np.asarray(vals, dtype=float)
        if arr.size == 0:
            return arr

        mu = float(arr.mean())
        sd = float(arr.std())
        if not np.isfinite(sd) or sd < 1e-6:
            sd = 1.0
        # Higher effect → better → negate to lower-is-better
        return -(arr - mu) / sd


def _safe_component(
    arr: "np.ndarray | None", n: int, name: str
) -> "np.ndarray":
    import numpy as np

    if arr is None:
        logger.info("[ensemble.combine] %s unavailable; using zeros", name)
        return np.zeros(n, dtype=float)
    if len(arr) != n:
        logger.info(
            "[ensemble.combine] %s length %d != %d; using zeros", name, len(arr), n
        )
        return np.zeros(n, dtype=float)
    return arr.astype(float)


def combine_pace(
    gbm_pace: "np.ndarray",
    elo_pace: "Optional[np.ndarray]",
    bt_pace: "Optional[np.ndarray]",
    mixed_pace: "Optional[np.ndarray]",
    cfg: Optional[EnsembleConfig] = None,
    session_type: str = "race",
) -> "np.ndarray":
    """Combine multiple pace components into a single pace index.

    - All inputs are assumed to be on a "lower is better" scale.
    - The weight set used depends on *session_type*: qualifying/sprint_qualifying
      sessions use the ``*_quali`` weights so that the race-only statistical
      models (Elo, BT, Mixed) are down-weighted relative to the GBM, which
      already switches its target to ``qualifying_form_index`` for those sessions.
    - If any component is missing or has wrong length, it is safely replaced
      with zeros and a debug log is emitted.
    """
    import numpy as np

    if gbm_pace is None:
        logger.info("[ensemble.combine] gbm_pace is None; returning empty array")
        return np.zeros(0, dtype=float)

    n = len(gbm_pace)
    if n == 0:
        return np.zeros(0, dtype=float)

    cfg = cfg or EnsembleConfig()

    g = _safe_component(gbm_pace, n, "gbm")
    e = _safe_component(elo_pace, n, "elo") if elo_pace is not None else np.zeros(n)
    b = _safe_component(bt_pace, n, "bt") if bt_pace is not None else np.zeros(n)
    m = _safe_component(mixed_pace, n, "mixed") if mixed_pace is not None else np.zeros(n)

    is_quali = session_type in _QUALI_SESSIONS
    if is_quali:
        w_gbm_raw = cfg.w_gbm_quali
        w_elo_raw = cfg.w_elo_quali
        w_bt_raw = cfg.w_bt_quali
        w_mixed_raw = cfg.w_mixed_quali
    else:
        w_gbm_raw = cfg.w_gbm
        w_elo_raw = cfg.w_elo
        w_bt_raw = cfg.w_bt
        w_mixed_raw = cfg.w_mixed

    wsum = w_gbm_raw + w_elo_raw + w_bt_raw + w_mixed_raw
    if wsum <= 0:
        logger.info("[ensemble.combine] Non-positive weight sum; falling back to gbm only")
        w_gbm_raw, w_elo_raw, w_bt_raw, w_mixed_raw = 1.0, 0.0, 0.0, 0.0
        wsum = 1.0

    w_gbm = w_gbm_raw / wsum
    w_elo = w_elo_raw / wsum
    w_bt = w_bt_raw / wsum
    w_mixed = w_mixed_raw / wsum

    combined = w_gbm * g + w_elo * e + w_bt * b + w_mixed * m

    mu = float(combined.mean())
    sd = float(combined.std())
    if not np.isfinite(sd) or sd < cfg.min_std:
        sd = 1.0
    z = (combined - mu) / sd

    # Deterministic tiny jitter to break exact ties reproducibly
    jitter = (np.arange(n, dtype=float) % 997) / 1e6
    out = z + jitter

    track = "quali" if is_quali else "race"
    logger.info(
        "[ensemble.combine] %s pace: std=%.6f, range=%.6f "
        "(w_gbm=%.2f w_elo=%.2f w_bt=%.2f w_mixed=%.2f)",
        track,
        float(out.std()),
        float(np.ptp(out)),
        w_gbm,
        w_elo,
        w_bt,
        w_mixed,
    )
    return out
