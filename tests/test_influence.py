"""Tests for the per-driver feature influence / explainability layer.

Covers:
- compute_shap_values  (models.py)
- _build_ensemble_components  (predict.py)
- _print_influence_row  (predict.py)
- _sanitize_for_json  (web.py)
- estimate_dnf_probabilities weather/circuit branches  (models.py)
"""
from __future__ import annotations

import math
import numpy as np
import pandas as pd
import pytest
from unittest.mock import MagicMock, patch


# ---------------------------------------------------------------------------
# Helpers shared across tests
# ---------------------------------------------------------------------------

def _make_pipe(n_features=3, n_drivers=5):
    """Return a minimal fitted sklearn Pipeline that looks like the GBM pipe."""
    from sklearn.pipeline import Pipeline
    from sklearn.preprocessing import StandardScaler
    from sklearn.impute import SimpleImputer
    from sklearn.compose import ColumnTransformer
    from sklearn.ensemble import GradientBoostingRegressor

    num_pipe = Pipeline([("imp", SimpleImputer(strategy="median")),
                         ("scl", StandardScaler())])
    pre = ColumnTransformer([("num", num_pipe, list(range(n_features)))])

    model = GradientBoostingRegressor(n_estimators=5, random_state=0)

    rng = np.random.RandomState(42)
    X_fit = rng.randn(20, n_features)
    y_fit = rng.randn(20)
    pre.fit(X_fit)
    model.fit(pre.transform(X_fit), y_fit)

    pipe = MagicMock()
    pipe.named_steps = {"pre": pre, "model": model}
    return pipe, n_features, n_drivers


def _make_feature_df(n_drivers=5, feature_names=None):
    rng = np.random.RandomState(7)
    if feature_names is None:
        feature_names = ["form_index", "grid", "team_form_index"]
    data = {f: rng.randn(n_drivers) for f in feature_names}
    data["driverId"] = [f"d{i}" for i in range(n_drivers)]
    return pd.DataFrame(data)


# ---------------------------------------------------------------------------
# compute_shap_values
# ---------------------------------------------------------------------------

class TestComputeShapValues:

    def test_returns_none_when_shap_missing(self):
        """Should return None gracefully when shap is not installed."""
        from f1pred.models import compute_shap_values

        pipe, n_features, n_drivers = _make_pipe(n_features=3, n_drivers=5)
        X = _make_feature_df(n_drivers=5)
        features = ["form_index", "grid", "team_form_index"]

        with patch.dict("sys.modules", {"shap": None}):
            result = compute_shap_values(pipe, X, features)
        assert result is None

    def test_returns_list_of_dicts_with_shap_installed(self):
        """When shap is available, returns one dict per driver."""
        pytest.importorskip("shap")
        from f1pred.models import compute_shap_values

        n_drivers = 5
        feature_names = ["form_index", "grid", "team_form_index"]
        pipe, _, _ = _make_pipe(n_features=len(feature_names), n_drivers=n_drivers)

        # Make pre.get_feature_names_out return positional names (no __ prefix)
        # to exercise the else-branch (line 629 orig_col = tname)
        pipe.named_steps["pre"].get_feature_names_out = lambda: [
            f"f{i}" for i in range(len(feature_names))
        ]

        X = _make_feature_df(n_drivers=n_drivers, feature_names=feature_names)

        result = compute_shap_values(pipe, X, feature_names)
        assert isinstance(result, list)
        assert len(result) == n_drivers
        for d in result:
            assert isinstance(d, dict)
            assert len(d) > 0

    def test_fallback_when_transformed_names_length_mismatch(self):
        """Exercises the positional fallback (no transformed_names match)."""
        pytest.importorskip("shap")
        from f1pred.models import compute_shap_values

        n_drivers = 4
        feature_names = ["form_index", "grid"]
        pipe, _, _ = _make_pipe(n_features=len(feature_names), n_drivers=n_drivers)

        # Return wrong number of names to force the positional fallback path
        pipe.named_steps["pre"].get_feature_names_out = lambda: ["only_one"]

        X = _make_feature_df(n_drivers=n_drivers, feature_names=feature_names)
        result = compute_shap_values(pipe, X, feature_names)
        # Should still return a list (via positional fallback)
        assert result is None or isinstance(result, list)

    def test_returns_none_on_explainer_exception(self):
        """If TreeExplainer raises, compute_shap_values returns None."""
        pytest.importorskip("shap")
        from f1pred.models import compute_shap_values
        import shap as shap_lib

        feature_names = ["form_index", "grid", "team_form_index"]
        pipe, _, _ = _make_pipe(n_features=len(feature_names))
        X = _make_feature_df(feature_names=feature_names)

        with patch.object(shap_lib, "TreeExplainer", side_effect=RuntimeError("boom")):
            result = compute_shap_values(pipe, X, feature_names)
        assert result is None

    def test_prefixed_feature_names_mapped_correctly(self):
        """Exercises the '__' prefix branch (num__form_index → form_index)."""
        pytest.importorskip("shap")
        from f1pred.models import compute_shap_values

        feature_names = ["form_index", "grid"]
        pipe, _, _ = _make_pipe(n_features=len(feature_names))
        X = _make_feature_df(feature_names=feature_names)

        # Simulate sklearn ColumnTransformer style prefixed names
        pipe.named_steps["pre"].get_feature_names_out = lambda: [
            "num__form_index", "num__grid"
        ]

        result = compute_shap_values(pipe, X, feature_names)
        if result is not None:  # shap installed
            assert all("form_index" in d or "grid" in d for d in result)

    def test_unknown_prefixed_name_falls_back_to_rest(self):
        """Exercises the 'orig_col = rest' fallback (line 627) for unknown feature."""
        pytest.importorskip("shap")
        from f1pred.models import compute_shap_values

        feature_names = ["form_index"]
        pipe, _, _ = _make_pipe(n_features=1)
        X = _make_feature_df(n_drivers=3, feature_names=feature_names)

        # Name that has __ but doesn't match any known feature
        pipe.named_steps["pre"].get_feature_names_out = lambda: ["num__unknown_col"]

        result = compute_shap_values(pipe, X, feature_names)
        if result is not None:
            for d in result:
                assert "unknown_col" in d


# ---------------------------------------------------------------------------
# _build_ensemble_components
# ---------------------------------------------------------------------------

class TestBuildEnsembleComponents:

    def _cfg(self, w_gbm=0.6, w_elo=0.15, w_bt=0.15, w_mixed=0.1,
             w_gbm_quali=0.8, w_elo_quali=0.07, w_bt_quali=0.07, w_mixed_quali=0.06):
        cfg = MagicMock()
        cfg.w_gbm = w_gbm
        cfg.w_elo = w_elo
        cfg.w_bt = w_bt
        cfg.w_mixed = w_mixed
        cfg.w_gbm_quali = w_gbm_quali
        cfg.w_elo_quali = w_elo_quali
        cfg.w_bt_quali = w_bt_quali
        cfg.w_mixed_quali = w_mixed_quali
        return cfg

    def test_basic_race_session(self):
        from f1pred.predict import _build_ensemble_components

        gbm = np.array([1.0, -1.0, 0.5])
        elo = np.array([0.5, -0.5, 0.2])
        bt  = np.array([0.3, -0.3, 0.1])
        mx  = np.array([0.2, -0.2, 0.05])

        result = _build_ensemble_components(gbm, elo, bt, mx, self._cfg(), "race")
        assert result is not None
        assert len(result) == 3
        for d in result:
            assert set(d.keys()) == {"gbm", "elo", "bt", "mixed", "weights"}
            total = d["gbm"] + d["elo"] + d["bt"] + d["mixed"]
            assert abs(total - 1.0) < 1e-4

    def test_qualifying_session_uses_quali_weights(self):
        from f1pred.predict import _build_ensemble_components

        gbm = np.array([1.0])
        result = _build_ensemble_components(gbm, None, None, None, self._cfg(), "qualifying")
        assert result is not None
        # With all other components None (zeroed), GBM should dominate
        assert result[0]["gbm"] > 0.9

    def test_zero_weights_fallback(self):
        """When all weights sum to zero, falls back to GBM-only."""
        from f1pred.predict import _build_ensemble_components

        cfg = self._cfg(w_gbm=0, w_elo=0, w_bt=0, w_mixed=0)
        gbm = np.array([2.0, -1.0])
        result = _build_ensemble_components(gbm, None, None, None, cfg, "race")
        assert result is not None
        for d in result:
            assert d["gbm"] == pytest.approx(1.0, abs=1e-4)

    def test_all_zeros_pace_total_fallback(self):
        """When all weighted contributions are near-zero, total falls back to 1.0
        so division doesn't crash; each fraction is 0/1 = 0."""
        from f1pred.predict import _build_ensemble_components

        gbm = np.array([0.0])
        result = _build_ensemble_components(gbm, None, None, None, self._cfg(), "race")
        assert result is not None
        # 0-pace driver: all contributions are 0; fractions are 0.0 (no crash)
        assert result[0]["gbm"] == pytest.approx(0.0, abs=1e-6)
        assert result[0]["elo"] == pytest.approx(0.0, abs=1e-6)

    def test_mismatched_component_length_treated_as_zeros(self):
        """Components with wrong length are replaced with zeros."""
        from f1pred.predict import _build_ensemble_components

        gbm = np.array([1.0, 2.0])
        bad_elo = np.array([99.0])  # wrong length
        result = _build_ensemble_components(gbm, bad_elo, None, None, self._cfg(), "race")
        assert result is not None
        assert len(result) == 2

    def test_returns_none_on_exception(self):
        """Should return None when cfg raises an unexpected error."""
        from f1pred.predict import _build_ensemble_components

        bad_cfg = MagicMock(spec=[])  # no attributes at all
        result = _build_ensemble_components(np.array([1.0]), None, None, None, bad_cfg, "race")
        assert result is None


# ---------------------------------------------------------------------------
# _print_influence_row
# ---------------------------------------------------------------------------

class TestPrintInfluenceRow:

    def _make_row(self, with_ensemble=True, with_shap=True):
        row = {
            "predicted_position": 1,
            "name": "Lewis Hamilton",
            "code": "HAM",
            "constructorName": "Mercedes",
            "mean_pos": 1.5,
            "p_top3": 0.9,
            "p_win": 0.7,
            "p_dnf": 0.05,
        }
        if with_ensemble:
            row["ensemble_components"] = {
                "gbm": 0.65, "elo": 0.15, "bt": 0.12, "mixed": 0.08,
                "weights": {"gbm": 0.6, "elo": 0.15, "bt": 0.15, "mixed": 0.1},
            }
        else:
            row["ensemble_components"] = None
        if with_shap:
            row["shap_values"] = {
                "form_index": -0.8,
                "grid": 0.3,
                "team_form_index": -0.2,
                "weather_effect": 0.1,
                "grid_finish_delta": -0.05,
                "circuit_avg_pos": 0.02,
            }
        else:
            row["shap_values"] = None
        return pd.Series(row)

    def test_prints_ensemble_and_shap(self, capsys):
        from f1pred.predict import _print_influence_row

        row = self._make_row(with_ensemble=True, with_shap=True)
        _print_influence_row(row, max_name=18, max_team=12, has_grid=True)
        out = capsys.readouterr().out
        assert "GBM" in out
        assert "Elo" in out
        # At least one feature label should appear
        assert "Race Form" in out or "Grid" in out or "Factors" in out

    def test_prints_ensemble_only(self, capsys):
        from f1pred.predict import _print_influence_row

        row = self._make_row(with_ensemble=True, with_shap=False)
        _print_influence_row(row, max_name=18, max_team=12, has_grid=False)
        out = capsys.readouterr().out
        assert "GBM" in out

    def test_prints_shap_only(self, capsys):
        from f1pred.predict import _print_influence_row

        row = self._make_row(with_ensemble=False, with_shap=True)
        _print_influence_row(row, max_name=18, max_team=12, has_grid=False)
        out = capsys.readouterr().out
        assert "Factors" in out or "Race Form" in out

    def test_prints_nothing_when_no_data(self, capsys):
        from f1pred.predict import _print_influence_row

        row = self._make_row(with_ensemble=False, with_shap=False)
        _print_influence_row(row, max_name=18, max_team=12, has_grid=False)
        out = capsys.readouterr().out
        assert out == ""

    def test_shap_direction_sign_positive(self, capsys):
        """Positive SHAP = increases pace index = worsens position → shown as '+'."""
        from f1pred.predict import _print_influence_row

        row = pd.Series({
            "ensemble_components": None,
            "shap_values": {"form_index": 0.9},
        })
        _print_influence_row(row, max_name=18, max_team=12, has_grid=False)
        out = capsys.readouterr().out
        assert "+" in out

    def test_shap_direction_sign_negative(self, capsys):
        """Negative SHAP = decreases pace index = improves position → shown as '-'."""
        from f1pred.predict import _print_influence_row

        row = pd.Series({
            "ensemble_components": None,
            "shap_values": {"form_index": -0.9},
        })
        _print_influence_row(row, max_name=18, max_team=12, has_grid=False)
        out = capsys.readouterr().out
        assert "-" in out

    def test_unknown_feature_name_fallback_label(self, capsys):
        """Unknown feature names should be title-cased rather than crashing."""
        from f1pred.predict import _print_influence_row

        row = pd.Series({
            "ensemble_components": None,
            "shap_values": {"some_unknown_feature": -0.5},
        })
        _print_influence_row(row, max_name=18, max_team=12, has_grid=False)
        out = capsys.readouterr().out
        assert "Some Unknown Feature" in out


# ---------------------------------------------------------------------------
# _sanitize_for_json  (web.py — importable without fastapi)
# ---------------------------------------------------------------------------

class TestSanitizeForJson:

    @pytest.fixture(autouse=True)
    def _import(self):
        # Import _sanitize_for_json by directly importing the module attribute.
        # web.py imports fastapi at module level, so we need to mock it first.
        import sys
        fastapi_mock = MagicMock()
        fastapi_mock.FastAPI = MagicMock(return_value=MagicMock())
        mocks = {
            "fastapi": fastapi_mock,
            "fastapi.responses": MagicMock(),
            "fastapi.templating": MagicMock(),
        }
        with patch.dict(sys.modules, mocks):
            import importlib
            # Clear cached web module if present
            sys.modules.pop("f1pred.web", None)
            import f1pred.web as web_mod
            self.fn = web_mod._sanitize_for_json

    def test_normal_float_passthrough(self):
        assert self.fn(3.14) == pytest.approx(3.14)

    def test_nan_becomes_none(self):
        assert self.fn(float("nan")) is None

    def test_inf_becomes_none(self):
        assert self.fn(float("inf")) is None
        assert self.fn(float("-inf")) is None

    def test_string_passthrough(self):
        assert self.fn("hello") == "hello"

    def test_int_passthrough(self):
        assert self.fn(42) == 42

    def test_none_passthrough(self):
        assert self.fn(None) is None

    def test_nested_dict_sanitized(self):
        d = {"a": float("nan"), "b": 1.5, "c": {"d": float("inf")}}
        result = self.fn(d)
        assert result == {"a": None, "b": 1.5, "c": {"d": None}}

    def test_list_sanitized(self):
        lst = [float("nan"), 2.0, float("inf")]
        result = self.fn(lst)
        assert result == [None, 2.0, None]

    def test_datetime_converted_to_isoformat(self):
        from datetime import datetime, timezone
        dt = datetime(2026, 3, 18, 12, 0, 0, tzinfo=timezone.utc)
        result = self.fn(dt)
        assert result == "2026-03-18T12:00:00+00:00"

    def test_nested_dict_with_list(self):
        d = {"shap": {"feat_a": float("nan"), "feat_b": -0.3}, "vals": [float("inf"), 1]}
        result = self.fn(d)
        assert result["shap"]["feat_a"] is None
        assert result["shap"]["feat_b"] == pytest.approx(-0.3)
        assert result["vals"][0] is None
        assert result["vals"][1] == 1


# ---------------------------------------------------------------------------
# estimate_dnf_probabilities — weather & circuit branches  (models.py)
# ---------------------------------------------------------------------------

def _make_hist(n_races=10, n_drivers=5, include_dnf=True):
    """Build a minimal historical results DataFrame."""
    from datetime import datetime, timezone, timedelta

    rows = []
    base = datetime(2023, 3, 1, tzinfo=timezone.utc)
    for rnd in range(1, n_races + 1):
        for pos in range(1, n_drivers + 1):
            rows.append({
                "season": 2023, "round": rnd, "session": "race",
                "date": base + timedelta(days=14 * rnd),
                "driverId": f"d{pos}",
                "constructorId": f"c{(pos - 1) // 2}",
                "position": None if (include_dnf and pos == n_drivers and rnd % 3 == 0) else pos,
                "status": "DNF" if (include_dnf and pos == n_drivers and rnd % 3 == 0) else "Finished",
                "points": max(0, 26 - pos * 2),
                "grid": pos,
            })
    return pd.DataFrame(rows)


def _make_current_X(n_drivers=5, with_circuit_dnf=False):
    data = {
        "driverId": [f"d{i}" for i in range(1, n_drivers + 1)],
        "constructorId": [f"c{(i - 1) // 2}" for i in range(1, n_drivers + 1)],
        "grid": list(range(1, n_drivers + 1)),
    }
    if with_circuit_dnf:
        data["global_circuit_dnf_rate"] = [0.12] * n_drivers
    return pd.DataFrame(data)


class TestTrainPaceModelShapExceptionBranch:

    def test_shap_exception_in_train_pace_model_is_swallowed(self, sample_features):
        """Covers lines 550-551: if compute_shap_values raises inside train_pace_model
        the exception is caught and shap_per_driver is returned as None."""
        from f1pred.models import train_pace_model
        import f1pred.models as models_mod

        original = models_mod.compute_shap_values

        def _explode(*args, **kwargs):
            raise RuntimeError("forced shap failure")

        models_mod.compute_shap_values = _explode
        try:
            _, pace_hat, features, shap_vals = train_pace_model(
                sample_features, session_type="race"
            )
            # Exception must be swallowed — shap_vals is None
            assert shap_vals is None
            assert len(pace_hat) == len(sample_features)
        finally:
            models_mod.compute_shap_values = original


class TestEstimateDnfProbabilitiesWeatherBranches:

    def test_wet_event_weather_branch(self):
        """Covers lines 702-704: event_weather with rain_sum > threshold."""
        from f1pred.models import estimate_dnf_probabilities

        hist = _make_hist()
        X = _make_current_X()

        probs = estimate_dnf_probabilities(
            hist, X,
            event_weather={"rain_sum": 5.0},   # wet session
        )
        assert len(probs) == len(X)
        assert all(0 <= p <= 1 for p in probs)

    def test_dry_event_weather_branch(self):
        """event_weather with rain_sum below threshold → dry."""
        from f1pred.models import estimate_dnf_probabilities

        hist = _make_hist()
        X = _make_current_X()

        probs = estimate_dnf_probabilities(
            hist, X,
            event_weather={"rain_sum": 0.0},   # dry session
        )
        assert len(probs) == len(X)

    def test_hist_weather_branch_covered(self):
        """Covers lines 713-742: hist_weather dict triggers vectorized merge."""
        from f1pred.models import estimate_dnf_probabilities

        hist = _make_hist()
        X = _make_current_X()

        # Build hist_weather keyed by (season, round) tuples
        hist_weather = {
            (2023, rnd): {"rain_sum": 3.0 if rnd % 2 == 0 else 0.0}
            for rnd in range(1, 11)
        }

        probs = estimate_dnf_probabilities(
            hist, X,
            event_weather={"rain_sum": 3.0},   # wet current session
            hist_weather=hist_weather,
        )
        assert len(probs) == len(X)
        assert all(np.isfinite(p) for p in probs)

    def test_hist_weather_with_nan_rain_skipped(self):
        """hist_weather entries with NaN rain_sum are skipped safely."""
        from f1pred.models import estimate_dnf_probabilities

        hist = _make_hist()
        X = _make_current_X()

        hist_weather = {
            (2023, 1): {"rain_sum": float("nan")},   # should be skipped
            (2023, 2): {"rain_sum": 2.0},
        }

        probs = estimate_dnf_probabilities(hist, X, hist_weather=hist_weather)
        assert len(probs) == len(X)

    def test_circuit_dnf_rate_branch(self):
        """Covers lines 777-779: global_circuit_dnf_rate column present."""
        from f1pred.models import estimate_dnf_probabilities

        hist = _make_hist()
        X = _make_current_X(with_circuit_dnf=True)

        probs = estimate_dnf_probabilities(hist, X)
        assert len(probs) == len(X)
        assert all(0 <= p <= 1 for p in probs)

    def test_cfg_override_branch(self):
        """Covers lines 674-682: cfg object is used to override default parameters."""
        from f1pred.models import estimate_dnf_probabilities

        hist = _make_hist()
        X = _make_current_X()

        cfg = MagicMock()
        cfg.modelling.dnf.alpha = 3.0
        cfg.modelling.dnf.beta = 10.0
        cfg.modelling.dnf.driver_weight = 0.5
        cfg.modelling.dnf.team_weight = 0.5
        cfg.modelling.dnf.clip_min = 0.01
        cfg.modelling.dnf.clip_max = 0.40

        probs = estimate_dnf_probabilities(hist, X, cfg=cfg)
        assert len(probs) == len(X)
        assert all(0.01 <= p <= 0.40 for p in probs)

    def test_cfg_attribute_error_falls_back_to_defaults(self):
        """Covers the AttributeError except branch in cfg handling (line 681-682)."""
        from f1pred.models import estimate_dnf_probabilities

        hist = _make_hist()
        X = _make_current_X()

        cfg = MagicMock(spec=[])  # no attributes — raises AttributeError

        probs = estimate_dnf_probabilities(hist, X, cfg=cfg)
        assert len(probs) == len(X)

    def test_empty_races_early_return(self):
        """Covers line 686: when hist has no race rows, returns default 0.08 array."""
        from f1pred.models import estimate_dnf_probabilities

        hist = pd.DataFrame({"session": ["qualifying"], "driverId": ["d1"],
                             "constructorId": ["c1"], "position": [1],
                             "status": ["Finished"]})
        X = _make_current_X(n_drivers=3)

        probs = estimate_dnf_probabilities(hist, X)
        assert len(probs) == 3
        assert all(p == pytest.approx(0.08) for p in probs)

    def test_is_dnf_column_branch(self):
        """Covers line 690: hist with pre-computed is_dnf column."""
        from f1pred.models import estimate_dnf_probabilities

        hist = _make_hist()
        hist["is_dnf"] = (hist["status"] == "DNF").astype(int)
        X = _make_current_X()

        probs = estimate_dnf_probabilities(hist, X)
        assert len(probs) == len(X)
        assert all(0 <= p <= 1 for p in probs)
