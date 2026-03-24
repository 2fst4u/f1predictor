"""Tests for the PredictionManager and diff computation."""
import pytest
from unittest.mock import patch, MagicMock
from f1pred.prediction_manager import (
    PredictionManager,
    PredictionDiff,
    DriverMovement,
    compute_prediction_diff,
    _fingerprint_predictions,
)


class TestDriverMovement:
    def test_to_dict(self):
        m = DriverMovement(
            driver_id="ver01",
            driver_name="Max Verstappen",
            code="VER",
            team="Red Bull",
            old_position=3,
            new_position=1,
            direction=2,
            reasons=["Weather Impact improved"],
        )
        d = m.to_dict()
        assert d["driver_id"] == "ver01"
        assert d["new_position"] == 1
        assert d["direction"] == 2
        assert "Weather Impact improved" in d["reasons"]


class TestPredictionDiff:
    def test_to_dict_empty(self):
        diff = PredictionDiff(session="race")
        d = diff.to_dict()
        assert d["session"] == "race"
        assert d["movements"] == []
        assert d["timestamp"]  # Auto-populated

    def test_to_dict_with_movements(self):
        m = DriverMovement("ver01", "Max Verstappen", "VER", "Red Bull", 2, 1, 1, ["Grid updated"])
        diff = PredictionDiff(session="race", movements=[m], changed_variables=["Grid updated"])
        d = diff.to_dict()
        assert len(d["movements"]) == 1
        assert d["movements"][0]["driver_id"] == "ver01"


class TestComputePredictionDiff:
    def _make_pred(self, driver_id, pos, code="???", name="Driver", team="Team",
                   grid=None, shap=None, form_index=None):
        p = {
            "driverId": driver_id,
            "predicted_position": pos,
            "code": code,
            "name": name,
            "constructorName": team,
            "p_win": 0.1,
            "p_top3": 0.3,
            "mean_pos": float(pos),
        }
        if grid is not None:
            p["grid"] = grid
        if shap is not None:
            p["shap_values"] = shap
        if form_index is not None:
            p["form_index"] = form_index
        return p

    def test_no_diff_same_order(self):
        old = [self._make_pred("d1", 1), self._make_pred("d2", 2)]
        new = [self._make_pred("d1", 1), self._make_pred("d2", 2)]
        result = compute_prediction_diff("race", old, new)
        assert result is None

    def test_diff_when_positions_swap(self):
        old = [
            self._make_pred("ver01", 1, "VER", "Max Verstappen", "Red Bull"),
            self._make_pred("ham44", 2, "HAM", "Lewis Hamilton", "Mercedes"),
        ]
        new = [
            self._make_pred("ham44", 1, "HAM", "Lewis Hamilton", "Mercedes"),
            self._make_pred("ver01", 2, "VER", "Max Verstappen", "Red Bull"),
        ]
        result = compute_prediction_diff("race", old, new)
        assert result is not None
        assert result.session == "race"
        assert len(result.movements) == 2

        # Find VER's movement
        ver_move = next(m for m in result.movements if m.driver_id == "ver01")
        assert ver_move.old_position == 1
        assert ver_move.new_position == 2
        assert ver_move.direction == -1  # Moved down

        ham_move = next(m for m in result.movements if m.driver_id == "ham44")
        assert ham_move.old_position == 2
        assert ham_move.new_position == 1
        assert ham_move.direction == 1  # Moved up

    def test_diff_detects_weather_change(self):
        old = [self._make_pred("d1", 1), self._make_pred("d2", 2)]
        new = [self._make_pred("d1", 2), self._make_pred("d2", 1)]
        old_weather = {"temp_mean": 25.0, "rain_sum": 0.0, "wind_mean": 10.0}
        new_weather = {"temp_mean": 25.0, "rain_sum": 5.0, "wind_mean": 10.0}

        result = compute_prediction_diff("race", old, new, old_weather, new_weather)
        assert result is not None
        assert "Weather forecast updated" in result.changed_variables

    def test_diff_detects_grid_change(self):
        old = [self._make_pred("d1", 1, grid=1), self._make_pred("d2", 2, grid=2)]
        new = [self._make_pred("d1", 2, grid=5), self._make_pred("d2", 1, grid=1)]
        result = compute_prediction_diff("race", old, new)
        assert result is not None
        assert "Grid positions updated (session results available)" in result.changed_variables

    def test_diff_with_shap_reasons(self):
        old = [
            self._make_pred("d1", 1, shap={"weather_effect": -0.5, "form_index": -1.0}),
            self._make_pred("d2", 2, shap={"weather_effect": 0.2, "form_index": -0.8}),
        ]
        new = [
            self._make_pred("d1", 2, shap={"weather_effect": 0.3, "form_index": -1.0}),
            self._make_pred("d2", 1, shap={"weather_effect": -0.6, "form_index": -0.8}),
        ]
        result = compute_prediction_diff("race", old, new)
        assert result is not None
        # d1 moved from P1 to P2, weather_effect changed from -0.5 to 0.3 (worsened)
        d1_move = next(m for m in result.movements if m.driver_id == "d1")
        assert any("Weather Effect" in r for r in d1_move.reasons)

    def test_returns_none_for_empty_inputs(self):
        assert compute_prediction_diff("race", [], []) is None
        assert compute_prediction_diff("race", None, None) is None

    def test_movements_sorted_by_magnitude(self):
        old = [
            self._make_pred("d1", 1),
            self._make_pred("d2", 5),
            self._make_pred("d3", 3),
        ]
        new = [
            self._make_pred("d1", 2),   # Moved 1 place
            self._make_pred("d2", 2),   # Moved 3 places
            self._make_pred("d3", 5),   # Moved 2 places
        ]
        result = compute_prediction_diff("race", old, new)
        assert result is not None
        # Should be sorted by magnitude: d2 (3), d3 (2), d1 (1)
        assert result.movements[0].driver_id == "d2"
        assert result.movements[1].driver_id == "d3"
        assert result.movements[2].driver_id == "d1"


class TestFingerprintPredictions:
    def test_same_data_same_hash(self):
        p1 = [{"driverId": "d1", "predicted_position": 1, "p_win": 0.5, "p_top3": 0.8, "mean_pos": 1.5}]
        p2 = [{"driverId": "d1", "predicted_position": 1, "p_win": 0.5, "p_top3": 0.8, "mean_pos": 1.5}]
        assert _fingerprint_predictions(p1) == _fingerprint_predictions(p2)

    def test_different_data_different_hash(self):
        p1 = [{"driverId": "d1", "predicted_position": 1, "p_win": 0.5, "p_top3": 0.8, "mean_pos": 1.5}]
        p2 = [{"driverId": "d1", "predicted_position": 2, "p_win": 0.3, "p_top3": 0.6, "mean_pos": 2.5}]
        assert _fingerprint_predictions(p1) != _fingerprint_predictions(p2)

    def test_order_independent(self):
        """Hash should be stable regardless of list order (sorted by driverId)."""
        p1 = [
            {"driverId": "d1", "predicted_position": 1, "p_win": 0.5, "p_top3": 0.8, "mean_pos": 1.5},
            {"driverId": "d2", "predicted_position": 2, "p_win": 0.3, "p_top3": 0.6, "mean_pos": 2.5},
        ]
        p2 = [
            {"driverId": "d2", "predicted_position": 2, "p_win": 0.3, "p_top3": 0.6, "mean_pos": 2.5},
            {"driverId": "d1", "predicted_position": 1, "p_win": 0.5, "p_top3": 0.8, "mean_pos": 1.5},
        ]
        assert _fingerprint_predictions(p1) == _fingerprint_predictions(p2)


class TestPredictionManagerLifecycle:
    def test_start_stop(self):
        cfg = MagicMock()
        cfg.data_sources.jolpica.base_url = "https://api.example.com"
        cfg.data_sources.jolpica.timeout_seconds = 10
        cfg.data_sources.jolpica.rate_limit_sleep = 0.5
        cfg.modelling.targets.session_types = ["qualifying", "race"]

        manager = PredictionManager(cfg, poll_interval=60)
        assert manager.status == "idle"
        assert manager.latest_results is None
        assert manager.latest_diffs == []

        # Start should not raise
        manager.start()
        assert manager._running is True

        # Stop should be clean
        manager.stop()
        assert manager._running is False

    def test_min_poll_interval_enforced(self):
        cfg = MagicMock()
        manager = PredictionManager(cfg, poll_interval=10)
        assert manager.poll_interval == 60  # Minimum enforced

    def test_subscribe_unsubscribe(self):
        import asyncio
        cfg = MagicMock()
        manager = PredictionManager(cfg, poll_interval=60)

        q = manager.subscribe()
        assert isinstance(q, asyncio.Queue)
        assert len(manager._subscribers) == 1

        manager.unsubscribe(q)
        assert len(manager._subscribers) == 0

    def test_broadcast_to_subscribers(self):
        import asyncio
        cfg = MagicMock()
        manager = PredictionManager(cfg, poll_interval=60)

        q = manager.subscribe()
        event = {"type": "test", "message": "hello"}
        manager._broadcast(event)

        assert not q.empty()
        received = q.get_nowait()
        assert received["type"] == "test"
        assert received["message"] == "hello"
