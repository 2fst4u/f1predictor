"""Tests for the prediction-audit fixes.

Covers the new shared helpers and behaviours introduced by the audit:
finish-status DNF detection, weather anomaly normalization, form sufficient
statistics, the event weather map, simulation noise (incl. teammate
correlation), outcome-target GBM training, pit-lane grid parsing, and
partial-actual metric masking.
"""
import json
import os
from datetime import datetime, timedelta, timezone

import numpy as np
import pandas as pd
import pytest

from f1pred.features import (
    _parse_races_block,
    compute_dnf_flags,
    compute_form_components,
    compute_form_indices,
    get_event_weather_map,
    normalize_weather_conditions,
)
from f1pred.metrics import compute_event_metrics
from f1pred.models import build_hist_training_X, train_pace_model
from f1pred.simulate import noise_sigma, simulate_grid

UTC = timezone.utc
REF = datetime(2024, 6, 1, tzinfo=UTC)


# ---------------------------------------------------------------------------
# compute_dnf_flags — finish-status based DNF detection
# ---------------------------------------------------------------------------
class TestComputeDnfFlags:
    def test_finished_and_lapped_are_finishes(self):
        df = pd.DataFrame({
            "position": [1, 2, 3, 4],
            "status": ["Finished", "+1 Lap", "+2 Laps", "Lapped"],
        })
        assert not compute_dnf_flags(df).any()

    @pytest.mark.parametrize("status", [
        "Retired", "Puncture", "Overheating", "Power Unit", "Withdrew",
        "Engine", "Accident", "Wheel", "Driveshaft", "Transmission",
        "Disqualified", "DNF",
    ])
    def test_retirement_causes_are_dnf(self, status):
        df = pd.DataFrame({"position": [10], "status": [status]})
        assert compute_dnf_flags(df).all(), f"{status} should count as a non-finish"

    def test_missing_status_with_position_is_finish(self):
        df = pd.DataFrame({"position": [5, 6], "status": [None, np.nan]})
        assert not compute_dnf_flags(df).any()

    def test_missing_position_is_dnf(self):
        df = pd.DataFrame({"position": [np.nan], "status": ["Finished"]})
        assert compute_dnf_flags(df).all()

    def test_no_status_column_falls_back_to_position(self):
        df = pd.DataFrame({"position": [1, np.nan]})
        flags = compute_dnf_flags(df)
        assert list(flags) == [False, True]


# ---------------------------------------------------------------------------
# normalize_weather_conditions — comparable anomaly scales
# ---------------------------------------------------------------------------
class TestNormalizeWeather:
    def test_missing_values_are_neutral(self):
        assert normalize_weather_conditions(None, None, None, None) == (0.0, 0.0, 0.0, 0.0)
        assert normalize_weather_conditions(np.nan, np.nan, np.nan, np.nan) == (0.0, 0.0, 0.0, 0.0)

    def test_baseline_conditions_are_neutral(self):
        t, p, w, r = normalize_weather_conditions(22.0, 1013.0, 15.0, 0.0)
        assert (t, p, w, r) == (0.0, 0.0, 0.0, 0.0)

    def test_pressure_no_longer_dominates(self):
        # A typical pressure reading and a heavy-rain reading must land on
        # comparable scales (the old raw-unit formula gave pressure ~1013 vs
        # rain ~10 — two orders of magnitude apart).
        _, p_anom, _, r_anom = normalize_weather_conditions(22.0, 1020.0, 15.0, 10.0)
        assert abs(p_anom) < 5.0
        assert abs(r_anom) > 0.5
        assert abs(p_anom) / abs(r_anom) < 10.0

    def test_rain_is_clipped(self):
        _, _, _, r_extreme = normalize_weather_conditions(22.0, 1013.0, 15.0, 500.0)
        _, _, _, r_cap = normalize_weather_conditions(22.0, 1013.0, 15.0, 25.0)
        assert r_extreme == r_cap


# ---------------------------------------------------------------------------
# compute_form_components — exact ratio identity with compute_form_indices
# ---------------------------------------------------------------------------
class TestFormComponents:
    def _history(self):
        rows = []
        for rnd in range(1, 6):
            d = REF - timedelta(days=400 - rnd * 20)  # previous season
            rows.append({"season": 2023, "round": rnd, "session": "race", "date": d,
                         "driverId": "max", "position": rnd, "points": 10.0, "status": "Finished"})
        for rnd in range(1, 4):
            d = REF - timedelta(days=60 - rnd * 10)  # current season
            rows.append({"season": 2024, "round": rnd, "session": "race", "date": d,
                         "driverId": "max", "position": 10 + rnd, "points": 1.0, "status": "Finished"})
        return pd.DataFrame(rows)

    def test_ratio_matches_production_boost_formula(self):
        hist = self._history()
        comp = compute_form_components(hist, ref_date=REF, half_life_days=120, current_season=2024)
        row = comp[comp["driverId"] == "max"].iloc[0]

        for boost in (1.0, 8.0, 30.0):
            expected = compute_form_indices(
                hist, ref_date=REF, half_life_days=120,
                current_season=2024, boost_factor=boost, sprint_boost_factor=boost,
            )
            expected_val = expected.loc[expected["driverId"] == "max", "form_index"].iloc[0]
            # form(b_r, b_s) = (s_pre + b_r*s_cur_race + b_s*s_cur_sprint)
            #                / (w_pre + b_r*w_cur_race + b_s*w_cur_sprint)
            num = row.s_pre + boost * row.s_cur_race + boost * row.s_cur_sprint
            den = row.w_pre + boost * row.w_cur_race + boost * row.w_cur_sprint
            got = num / den
            assert got == pytest.approx(expected_val, rel=1e-9), f"boost={boost}"

    def test_race_and_sprint_buckets_are_separated(self):
        # A current-season sprint result lands in the sprint bucket, a race
        # result in the race bucket, so the two boosts can move independently.
        rows = [
            {"season": 2024, "round": 1, "session": "race",
             "date": REF - timedelta(days=20), "driverId": "max", "position": 2,
             "points": 18.0, "status": "Finished"},
            {"season": 2024, "round": 1, "session": "sprint",
             "date": REF - timedelta(days=21), "driverId": "max", "position": 5,
             "points": 4.0, "status": "Finished"},
        ]
        out = compute_form_components(
            pd.DataFrame(rows), ref_date=REF, half_life_days=120, current_season=2024,
        )
        row = out[out["driverId"] == "max"].iloc[0]
        assert row.w_cur_race > 0 and row.w_cur_sprint > 0
        assert row.s_cur_race / row.w_cur_race == pytest.approx(-2.0)
        assert row.s_cur_sprint / row.w_cur_sprint == pytest.approx(-5.0)

    def test_empty_history(self):
        out = compute_form_components(pd.DataFrame(), ref_date=REF, half_life_days=120, current_season=2024)
        assert out.empty

    def test_qpos_sessions(self):
        rows = [{"season": 2024, "round": 1, "session": "qualifying",
                 "date": REF - timedelta(days=10), "driverId": "max", "qpos": 3}]
        out = compute_form_components(
            pd.DataFrame(rows), ref_date=REF, half_life_days=120, current_season=2024,
            sessions=("qualifying", "sprint_qualifying"), pos_col="qpos",
        )
        row = out[out["driverId"] == "max"].iloc[0]
        # A plain "qualifying" session is a non-sprint -> race bucket
        assert row.w_cur_race > 0 and row.w_pre == 0
        assert row.s_cur_race / row.w_cur_race == pytest.approx(-3.0)


# ---------------------------------------------------------------------------
# get_event_weather_map — cached weather lookup, no network
# ---------------------------------------------------------------------------
def test_get_event_weather_map_reads_disk_cache(tmp_path):
    wdir = tmp_path / "weather"
    wdir.mkdir()
    payload = {"temp_mean": 20.0, "rain_sum": 3.0}
    with open(wdir / "event_2024_5.json", "w") as f:
        json.dump(payload, f)

    out = get_event_weather_map(str(tmp_path), [(2024, 5), (2024, 6)])
    assert out[(2024, 5)]["rain_sum"] == 3.0
    assert (2024, 6) not in out  # missing events are simply absent

    assert get_event_weather_map(str(tmp_path), []) == {}


# ---------------------------------------------------------------------------
# simulate — noise sigma and teammate-correlated noise
# ---------------------------------------------------------------------------
class TestSimulationNoise:
    def test_noise_sigma_floor_and_scale(self):
        z = np.array([-1.0, 0.0, 1.0])
        assert noise_sigma(z, noise_factor=0.15, min_noise=0.05) > 0.05
        # min_noise floor binds when the factor is tiny
        assert noise_sigma(z, noise_factor=0.001, min_noise=0.2) == 0.2
        # degenerate inputs stay finite
        assert np.isfinite(noise_sigma(np.array([]), 0.15, 0.05))
        assert np.isfinite(noise_sigma(np.zeros(5), 0.15, 0.05))

    def test_team_correlation_correlates_teammates(self):
        # Equal-pace field: with fully shared team noise, teammates receive
        # identical perturbations, so their finishing positions are adjacent
        # in every draw.  With independent noise they frequently are not.
        n = 6
        pace = np.zeros(n)
        dnf = np.zeros(n)
        teams = np.array([0, 0, 1, 1, 2, 2])

        prob_c, mp_c, _ = simulate_grid(pace, dnf, draws=2000, random_seed=11,
                                        team_codes=teams, team_correlation=1.0)
        prob_i, mp_i, _ = simulate_grid(pace, dnf, draws=2000, random_seed=11,
                                        team_codes=teams, team_correlation=0.0)
        # Probabilities stay well-formed in both modes
        assert np.allclose(prob_c.sum(axis=1), 1.0, atol=0.01)
        assert np.allclose(prob_i.sum(axis=1), 1.0, atol=0.01)
        # And the correlation changes the joint outcome distribution
        assert not np.allclose(prob_c, prob_i)

    def test_team_codes_with_missing_team(self):
        pace = np.zeros(4)
        teams = np.array([0, 0, -1, -1])  # two drivers without a team
        prob, mean_pos, _ = simulate_grid(pace, np.zeros(4), draws=500,
                                          team_codes=teams, team_correlation=0.5)
        assert np.all(np.isfinite(mean_pos))
        assert np.allclose(prob.sum(axis=1), 1.0, atol=0.01)


# ---------------------------------------------------------------------------
# Outcome-target GBM training
# ---------------------------------------------------------------------------
class TestOutcomeTraining:
    def _hist_and_X(self):
        drivers = [f"d{i}" for i in range(10)]
        teams = [f"t{i // 2}" for i in range(10)]
        rows = []
        base = datetime(2024, 1, 1, tzinfo=UTC)
        for rnd in range(1, 13):
            d = base + timedelta(days=14 * rnd)
            for i, drv in enumerate(drivers):
                rows.append({"season": 2024, "round": rnd, "session": "race", "date": d,
                             "driverId": drv, "constructorId": teams[i], "position": i + 1,
                             "points": max(0, 20 - 2 * i), "grid": i + 1, "status": "Finished"})
                rows.append({"season": 2024, "round": rnd, "session": "qualifying",
                             "date": d - timedelta(days=1), "driverId": drv,
                             "constructorId": teams[i], "qpos": i + 1})
        hist = pd.DataFrame(rows)
        X = pd.DataFrame({
            "driverId": drivers, "constructorId": teams,
            "form_index": -np.arange(1, 11, dtype=float),
            "qualifying_form_index": -np.arange(1, 11, dtype=float),
            "grid": np.arange(1, 11, dtype=float),
        })
        return hist, X, base + timedelta(days=300)

    def test_hist_matrix_has_outcome_targets_and_quali_samples(self):
        hist, X, ref = self._hist_and_X()
        hx = build_hist_training_X(hist, X, ref)
        assert hx is not None
        assert "y_pace" in hx.columns
        assert hx["y_pace"].notna().all()
        # Both race and qualifying outcome samples are present
        assert (hx["is_race"] == 1).sum() > 0
        assert (hx["is_qualifying"] == 1).sum() > 0
        # Targets are z-scored within events: mean ~0 overall
        assert abs(float(hx["y_pace"].mean())) < 0.2
        # Qualifying samples carry no grid (imputed downstream)
        assert hx.loc[hx["is_qualifying"] == 1, "grid"].isna().all()

    def test_model_trained_on_outcomes_recovers_order(self):
        hist, X, ref = self._hist_and_X()
        hx = build_hist_training_X(hist, X, ref)
        for session in ("race", "qualifying"):
            _, pace, _, _ = train_pace_model(X, session_type=session, hist_X=hx)
            assert len(pace) == len(X)
            assert np.all(np.isfinite(pace))
        # Race path: better drivers (lower historical positions) -> faster pace
        _, pace, _, _ = train_pace_model(X, session_type="race", hist_X=hx)
        order = np.argsort(pace)
        assert np.mean(order[:3]) < np.mean(order[-3:])

    def test_legacy_path_without_y_pace_still_works(self):
        hist, X, ref = self._hist_and_X()
        hx = build_hist_training_X(hist, X, ref)
        legacy = hx.drop(columns=["y_pace"])
        _, pace, _, _ = train_pace_model(X, session_type="race", hist_X=legacy)
        assert len(pace) == len(X)
        assert np.all(np.isfinite(pace))

    def test_missing_session_flags_do_not_crash(self):
        hist, X, ref = self._hist_and_X()
        hx = build_hist_training_X(hist, X, ref)
        stripped = hx.drop(columns=["is_race", "is_qualifying", "is_sprint"])
        _, pace, _, _ = train_pace_model(X, session_type="race", hist_X=stripped)
        assert len(pace) == len(X)


# ---------------------------------------------------------------------------
# Pit-lane grid parsing
# ---------------------------------------------------------------------------
def test_parse_races_block_maps_pit_lane_grid_to_none():
    block = [{
        "season": "2023", "round": "1", "date": "2023-03-05", "time": "14:00:00Z",
        "Circuit": {"circuitName": "Bahrain", "circuitId": "bahrain",
                    "Location": {"lat": "26.0", "long": "50.5"}},
        "Results": [
            {"Driver": {"driverId": "a"}, "Constructor": {"constructorId": "x"},
             "grid": "0", "position": "12", "status": "Finished", "points": "0"},
            {"Driver": {"driverId": "b"}, "Constructor": {"constructorId": "y"},
             "grid": "3", "position": "1", "status": "Finished", "points": "25"},
        ],
    }]
    rows = _parse_races_block(block, "race", datetime(2024, 1, 1, tzinfo=UTC))
    by_drv = {r["driverId"]: r for r in rows}
    assert by_drv["a"]["grid"] is None  # pit-lane start, not P0
    assert by_drv["b"]["grid"] == 3


# ---------------------------------------------------------------------------
# Metrics: probability metrics tolerate partially missing actuals
# ---------------------------------------------------------------------------
def test_metrics_mask_missing_actuals():
    df = pd.DataFrame({
        "driver_id": ["a", "b", "c"],
        "predicted_position": [1, 2, 3],
        "actual_position": [1, 2, np.nan],
    })
    prob = np.array([
        [0.8, 0.15, 0.05],
        [0.15, 0.7, 0.15],
        [0.05, 0.15, 0.8],
    ])
    pairwise = np.array([
        [0.5, 0.9, 0.9],
        [0.1, 0.5, 0.8],
        [0.1, 0.2, 0.5],
    ])
    m = compute_event_metrics(df, prob, pairwise, "race", 2024, 1)
    # Previously a single missing actual nulled CRPS and Brier entirely
    assert np.isfinite(m["crps"])
    assert np.isfinite(m["brier_pairwise"])
    assert m["spearman"] == pytest.approx(1.0)


# ---------------------------------------------------------------------------
# predict helpers: team codes and cached historical weather
# ---------------------------------------------------------------------------
def test_team_codes_for_roster():
    from f1pred.predict import _team_codes_for
    X = pd.DataFrame({"constructorId": ["red_bull", "red_bull", "ferrari", None]})
    codes = _team_codes_for(X)
    assert codes[0] == codes[1]
    assert codes[0] != codes[2]
    assert codes[3] == -1
    assert _team_codes_for(pd.DataFrame({"foo": [1]})) is None


# ---------------------------------------------------------------------------
# Calibration: previously-static params are now on-the-fly calibrated
# ---------------------------------------------------------------------------
class TestCalibrationGrids:
    def test_param_vector_includes_formerly_static_params(self):
        import f1pred.calibrate as c
        assert c.N_PARAMS == 27
        assert len(c.PARAM_BOUNDS) == 27
        assert len(c.PARAM_DEFAULTS) == 27
        # Every default within its bound
        for i, (lo, hi) in enumerate(c.PARAM_BOUNDS):
            assert lo <= c.PARAM_DEFAULTS[i] <= hi
        # Interpolation bounds must stay inside their grids so brackets exist
        assert c.PARAM_BOUNDS[23][0] >= c.H_BASE_GRID[0]
        assert c.PARAM_BOUNDS[23][1] <= c.H_BASE_GRID[-1]
        assert c.PARAM_BOUNDS[24][0] >= c.H_TEAM_GRID[0]
        assert c.PARAM_BOUNDS[24][1] <= c.H_TEAM_GRID[-1]
        assert c.PARAM_BOUNDS[25][0] >= c.ELO_K_GRID[0]
        assert c.PARAM_BOUNDS[25][1] <= c.ELO_K_GRID[-1]

    def test_unpack_emits_calibrated_recency_elo_team_corr(self):
        import f1pred.calibrate as c
        w = list(c.PARAM_DEFAULTS)
        w[23] = 200.0   # half_life_base
        w[24] = 300.0   # half_life_team
        w[25] = 30.0    # elo_k
        w[26] = 0.5     # team_correlation
        w[22] = 12.0    # sprint weight
        out = c._unpack_weights(w)
        assert out["recency"]["half_life_base"] == pytest.approx(200.0)
        assert out["recency"]["half_life_team"] == pytest.approx(300.0)
        assert out["elo"]["k"] == pytest.approx(30.0)
        assert out["simulation"]["team_correlation"] == pytest.approx(0.5)
        assert out["blending"]["current_season_sprint_weight"] == pytest.approx(12.0)
        # pack round-trips length
        assert len(c._pack_weights(out)) == c.N_PARAMS

    def test_unpack_pads_short_vectors(self):
        import f1pred.calibrate as c
        # A legacy 22-length vector still unpacks (padded with defaults)
        out = c._unpack_weights(c.PARAM_DEFAULTS[:22])
        assert out["recency"]["half_life_base"] == pytest.approx(c.PARAM_DEFAULTS[23])
        assert out["simulation"]["team_correlation"] == pytest.approx(c.PARAM_DEFAULTS[26])

    def test_log_interp_coeffs_brackets_and_blends(self):
        from f1pred.calibrate import _log_interp_coeffs, H_BASE_GRID
        lo, hi, t = _log_interp_coeffs(H_BASE_GRID, H_BASE_GRID[0])
        assert lo == 0 and t == pytest.approx(0.0)
        lo, hi, t = _log_interp_coeffs(H_BASE_GRID, H_BASE_GRID[-1])
        assert hi == len(H_BASE_GRID) - 1 and t == pytest.approx(1.0)
        # A value at the geometric midpoint of a cell blends ~0.5
        mid = (H_BASE_GRID[0] * H_BASE_GRID[1]) ** 0.5
        lo, hi, t = _log_interp_coeffs(H_BASE_GRID, mid)
        assert (lo, hi) == (0, 1)
        assert t == pytest.approx(0.5, abs=1e-9)
        # Out-of-range clamps into the grid
        _, _, t_low = _log_interp_coeffs(H_BASE_GRID, 1.0)
        assert 0.0 <= t_low <= 1.0

    def test_fit_model_grids_shapes(self):
        from f1pred.calibrate import _fit_model_grids, H_TEAM_GRID, ELO_K_GRID
        drivers = [f"d{i}" for i in range(6)]
        teams = [f"t{i // 2}" for i in range(6)]
        rows = []
        for rnd in range(1, 6):
            d = REF - timedelta(days=20 * rnd)
            for pos, (drv, tm) in enumerate(zip(drivers, teams), 1):
                rows.append({"season": 2024, "round": rnd, "session": "race", "date": d,
                             "driverId": drv, "constructorId": tm, "position": pos,
                             "qpos": pos, "points": max(0, 26 - pos), "status": "Finished"})
        hist = pd.DataFrame(rows)
        X = pd.DataFrame({"driverId": drivers, "constructorId": teams})
        elo, bt, mixed = _fit_model_grids(hist, X, "race")
        assert len(elo) == len(H_TEAM_GRID)
        assert len(elo[0]) == len(ELO_K_GRID)
        assert len(elo[0][0]) == len(X)
        assert len(bt) == len(H_TEAM_GRID) and len(bt[0]) == len(X)
        assert len(mixed) == len(H_TEAM_GRID)

    def test_form_component_grids_cover_all_half_lives(self):
        from f1pred.calibrate import _form_component_grids, H_BASE_GRID
        rows = []
        for rnd in range(1, 5):
            d = REF - timedelta(days=30 * rnd)
            rows.append({"season": 2024, "round": rnd, "session": "race", "date": d,
                         "driverId": "max", "position": rnd, "points": 10.0, "status": "Finished"})
        grids = _form_component_grids(pd.DataFrame(rows), REF, 2024)
        assert len(grids) == len(H_BASE_GRID)
        # Each grid point maps the driver to a 6-tuple of components
        assert all("max" in g for g in grids)
        assert all(len(g["max"]) == 6 for g in grids)


def test_collect_hist_weather_reads_cache(tmp_path):
    from f1pred.predict import _collect_hist_weather

    class _Paths:
        cache_dir = str(tmp_path)

    class _Cfg:
        paths = _Paths()

    wdir = tmp_path / "weather"
    wdir.mkdir()
    with open(wdir / "event_2024_3.json", "w") as f:
        json.dump({"rain_sum": 5.0}, f)

    hist = pd.DataFrame({
        "season": [2024, 2024, 2010],
        "round": [3, 4, 1],
        "session": ["race", "race", "race"],
    })
    out = _collect_hist_weather(hist, 2024, _Cfg())
    assert out is not None and out[(2024, 3)]["rain_sum"] == 5.0
    # Seasons older than ~6 years are not probed
    assert (2010, 1) not in out

    assert _collect_hist_weather(pd.DataFrame(), 2024, _Cfg()) is None
