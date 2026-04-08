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
        assert "Grid positions changed" in result.changed_variables

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

    def test_handles_none_values(self):
        """Should not crash when expected fields are None."""
        p = [{"driverId": "d1", "predicted_position": 1, "p_win": None, "p_top3": None, "mean_pos": None}]
        try:
            _fingerprint_predictions(p)
        except TypeError as e:
            pytest.fail(f"_fingerprint_predictions crashed on None values: {e}")


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

class TestPredictionManagerCycle:
    def test_predict_round_success(self):
        cfg = MagicMock()
        cfg.data_sources.jolpica.base_url = "https://api.example.com"
        cfg.data_sources.jolpica.timeout_seconds = 10
        cfg.data_sources.jolpica.rate_limit_sleep = 0.5
        cfg.modelling.targets.session_types = ["qualifying", "race"]

        manager = PredictionManager(cfg, poll_interval=60)
        manager._latest_results = {"season": 2024, "rounds": {}}

        import pandas as pd
        df = pd.DataFrame([{
            "driverId": "ver01",
            "predicted_position": 1,
            "code": "VER",
            "name": "Max Verstappen",
            "constructorName": "Red Bull",
            "p_win": 0.9,
            "p_top3": 0.99,
            "mean_pos": 1.1,
            "grid": 1
        }])

        fake_results = {
            "season": 2024,
            "round": 1,
            "sessions": {
                "race": {
                    "ranked": df,
                    "meta": {"weather": {"temp_mean": 25.0}}
                }
            }
        }

        with patch('f1pred.predict.run_predictions_for_event') as mock_predict:
            mock_predict.return_value = fake_results
            jc = MagicMock()
            manager._predict_round(jc, 2024, 1, {"raceName": "Bahrain Grand Prix", "round": 1})

        assert "1" in manager.latest_results["rounds"]
        assert "race" in manager.latest_results["rounds"]["1"]["sessions"]

    def test_run_season_cycle_resolve_fails(self):
        cfg = MagicMock()
        manager = PredictionManager(cfg, poll_interval=60)

        with patch('f1pred.predict.resolve_event') as mock_resolve:
            with patch('f1pred.data.jolpica.JolpicaClient') as mock_jc:
                mock_jc.return_value.get_season_schedule.side_effect = Exception("schedule fetch failed")
                mock_resolve.side_effect = Exception("API error")

                manager._run_season_cycle(0)

        assert manager.status == "error"

    def test_run_season_cycle_success(self):
        cfg = MagicMock()
        manager = PredictionManager(cfg, poll_interval=60)
        manager._running = True

        with patch('f1pred.data.jolpica.JolpicaClient') as mock_jc_class:
            with patch('f1pred.predict.resolve_event') as mock_resolve:
                mock_jc = mock_jc_class.return_value
                # Mock resolve_event to return (current_season, next_round, current_event)
                mock_resolve.return_value = ("2024", 2, MagicMock())

                # Mock schedule
                mock_schedule = [
                    {"round": "1", "raceName": "Bahrain GP", "season": "2024"},
                    {"round": "2", "raceName": "Saudi GP", "season": "2024"}
                ]
                mock_jc.get_season_schedule.return_value = mock_schedule

                with patch.object(manager, '_predict_round') as mock_predict_round:
                    manager._run_season_cycle(0)

                    # Ensure _predict_round was called for the rounds
                    # In cycle 0, round 1 is not next_round so it checks _latest_results
                    # but _latest_results is populated directly with the loop
                    # Actually _latest_results might skip if not in rounds and cycle_count % 24 != 0 and has_data
                    # Round 1: cycle 0 % 24 == 0, so it will run even if has_data.
                    # Round 2: is_next == True, so it will run regardless.
                    assert mock_predict_round.call_count == 2

                    # Assert arguments for the first call
                    mock_predict_round.assert_any_call(mock_jc, "2024", 1, mock_schedule[0])
                    # Assert arguments for the second call
                    mock_predict_round.assert_any_call(mock_jc, "2024", 2, mock_schedule[1])

        assert manager.status == "idle"

    def test_predict_round_with_diffs(self):
        import pandas as pd

        cfg = MagicMock()
        cfg.modelling.targets.session_types = ["qualifying", "race"]
        cfg.paths.cache_dir = "cache"

        manager = PredictionManager(cfg, poll_interval=60)
        manager._latest_results = {"season": 2024, "rounds": {}}

        # Setup initial state
        df1 = pd.DataFrame([{"driverId": "ver01", "predicted_position": 1, "mean_pos": 1.1, "grid": 1, "p_win": 0.9, "p_top3": 0.9, "code": "VER", "name": "Max Verstappen", "constructorName": "Red Bull"},
                            {"driverId": "ham44", "predicted_position": 2, "mean_pos": 2.1, "grid": 2, "p_win": 0.1, "p_top3": 0.5, "code": "HAM", "name": "Lewis", "constructorName": "Mercedes"}])

        fake_results1 = {
            "season": 2024,
            "round": 1,
            "sessions": {
                "race": {"ranked": df1, "meta": {"weather": {"temp_mean": 25.0}}}
            }
        }

        with patch('f1pred.predict.run_predictions_for_event') as mock_predict:
            mock_predict.return_value = fake_results1
            jc = MagicMock()
            manager._predict_round(jc, 2024, 1, {"raceName": "Bahrain Grand Prix", "round": 1})

        assert len(manager.latest_diffs) == 0

        # Now change the results to trigger a diff
        df2 = pd.DataFrame([{"driverId": "ham44", "predicted_position": 1, "mean_pos": 1.2, "grid": 1, "p_win": 0.6, "p_top3": 0.9, "code": "HAM", "name": "Lewis", "constructorName": "Mercedes"},
                            {"driverId": "ver01", "predicted_position": 2, "mean_pos": 1.8, "grid": 2, "p_win": 0.4, "p_top3": 0.8, "code": "VER", "name": "Max Verstappen", "constructorName": "Red Bull"}])

        fake_results2 = {
            "season": 2024,
            "round": 1,
            "sessions": {
                "race": {"ranked": df2, "meta": {"weather": {"temp_mean": 30.0}}}
            }
        }

        with patch('f1pred.predict.run_predictions_for_event') as mock_predict:
            mock_predict.return_value = fake_results2
            if hasattr(manager, '_send_discord_webhook'):
                with patch.object(manager, '_send_discord_webhook') as mock_discord:
                    manager._predict_round(jc, 2024, 1, {"raceName": "Bahrain Grand Prix", "round": 1})
                    assert mock_discord.call_count > 0
            else:
                manager._predict_round(jc, 2024, 1, {"raceName": "Bahrain Grand Prix", "round": 1})

        assert len(manager.latest_diffs) > 0
        diff = manager.latest_diffs[-1]
        assert diff["session"] == "R1_race"
        assert len(diff["movements"]) == 2
        assert "Grid positions changed" in diff["changed_variables"]

    def test_run_loop(self):
        import threading
        import time

        cfg = MagicMock()
        manager = PredictionManager(cfg, poll_interval=1)

        # Patch the actual sleep so we don't wait 5s
        with patch('time.sleep') as mock_sleep:
            # Patch the cycle to stop the loop after one iteration
            with patch.object(manager, '_run_season_cycle') as mock_cycle:
                def side_effect(cycle_count):
                    manager._running = False
                mock_cycle.side_effect = side_effect

                manager._running = True
                manager._run_loop()

                assert mock_cycle.called

class TestPredictionManagerActualResults:
    def test_predict_round_actuals_and_sanitize(self):
        import pandas as pd
        import datetime
        from unittest.mock import MagicMock, patch
        from f1pred.prediction_manager import PredictionManager

        cfg = MagicMock()
        cfg.data_sources.jolpica.base_url = "https://api.example.com"
        cfg.data_sources.jolpica.timeout_seconds = 10
        cfg.data_sources.jolpica.rate_limit_sleep = 0.5
        cfg.modelling.targets.session_types = ["race", "qualifying", "sprint", "practice"]

        manager = PredictionManager(cfg, poll_interval=60)
        manager._latest_results = {"season": 2024, "rounds": {}}

        date_obj = datetime.datetime(2024, 1, 1)

        df_race = pd.DataFrame([{
            "driverId": "ver01",
            "predicted_position": 1,
            "mean_pos": 1.5,
            "p_win": 0.5,
            "p_top3": 0.5,
            "meta_list": [1, float("nan")],
            "meta_dict": {"k": float("inf")},
            "meta_date": date_obj,
            "grid": 1,
            "code": "VER",
            "name": "Max Verstappen",
            "constructorName": "Red Bull"
        }])

        fake_results = {
            "season": 2024,
            "round": 1,
            "sessions": {
                "race": {"ranked": df_race, "meta": {}},
                "qualifying": {"ranked": pd.DataFrame(), "meta": {}},
                "sprint": {"ranked": pd.DataFrame(), "meta": {}},
                "practice": {"ranked": pd.DataFrame(), "meta": {}}
            }
        }

        with patch('f1pred.predict.run_predictions_for_event') as mock_predict:
            mock_predict.return_value = fake_results
            jc = MagicMock()

            jc.get_race_results.return_value = [
                {"Driver": {"driverId": "ver01"}, "position": "1"},
                {"Driver": {"driverId": "ham44"}, "position": "invalid"}
            ]

            jc.get_qualifying_results.side_effect = Exception("API failed")
            jc.get_sprint_results.return_value = []

            manager._predict_round(jc, 2024, 1, {"raceName": "Bahrain Grand Prix", "round": 1})

        # Check results
        rounds = manager.latest_results.get("rounds", {})
        assert "1" in rounds
        r_session = rounds["1"]["sessions"]["race"]["predictions"]

        # Check sanitization
        first_row = r_session[0]
        assert first_row["meta_list"] == [1, None]
        assert first_row["meta_dict"] == {"k": None}
        assert first_row["meta_date"] == date_obj.isoformat()

        # Check actual results mapping
        assert first_row["actual_position"] == 1
        assert first_row["frozen"] is True


class TestPredictionManagerCycleExceptionHandling:
    def test_run_season_cycle_handles_exceptions(self):
        from unittest.mock import MagicMock, patch
        from f1pred.prediction_manager import PredictionManager

        cfg = MagicMock()
        manager = PredictionManager(cfg, poll_interval=60)

        # Test exception path
        with patch.object(manager, '_run_season_cycle') as mock_cycle:
            mock_cycle.side_effect = Exception("Test Exception")
            manager._running = True

            # To prevent infinite loop in test, change _running to False inside side_effect
            # but we can just test the exception block manually or use run_loop if patched

            # We want to test lines 384-388 which are inside _run_loop() when _run_season_cycle throws
            with patch('time.sleep') as mock_sleep:
                def run_loop_iteration(cycle_count):
                    # throw first time
                    if cycle_count == 0:
                        raise Exception("Simulated loop error")
                    else:
                        manager._running = False

                mock_cycle.side_effect = run_loop_iteration
                manager._run_loop()

            assert manager.status == "error"

class TestPredictionManagerPredictRoundSessions:
    def test_predict_round_sprint_session(self):
        import pandas as pd
        from unittest.mock import MagicMock, patch
        from f1pred.prediction_manager import PredictionManager

        cfg = MagicMock()
        cfg.modelling.targets.session_types = ["sprint", "sprint_qualifying", "practice"]
        manager = PredictionManager(cfg, poll_interval=60)
        manager._latest_results = {"season": 2024, "rounds": {}}

        fake_results = {
            "season": 2024,
            "round": 1,
            "sessions": {
                "sprint": {"ranked": pd.DataFrame([{"driverId": "ver", "predicted_position": 1}]), "meta": {}},
                "sprint_qualifying": {"ranked": pd.DataFrame([{"driverId": "ver", "predicted_position": 1}]), "meta": {}}
            }
        }

        with patch('f1pred.predict.run_predictions_for_event') as mock_predict:
            mock_predict.return_value = fake_results
            jc = MagicMock()

            # This triggers lines 473 and 475
            manager._predict_round(jc, 2024, 1, {"raceName": "Sprint GP", "round": 1, "Sprint": {}, "SprintQualifying": {}})

        rounds = manager.latest_results.get("rounds", {})
        assert "sprint" in rounds["1"]["sessions"]
        assert "sprint_qualifying" in rounds["1"]["sessions"]

    def test_predict_round_fallback_sessions(self):
        import pandas as pd
        from unittest.mock import MagicMock, patch
        from f1pred.prediction_manager import PredictionManager

        cfg = MagicMock()
        # Set a session type that isn't matched to anything, triggering empty sessions fallback (line 477)
        cfg.modelling.targets.session_types = ["invalid_session"]
        manager = PredictionManager(cfg, poll_interval=60)
        manager._latest_results = {"season": 2024, "rounds": {}}

        fake_results = {
            "season": 2024,
            "round": 1,
            "sessions": {}
        }

        with patch('f1pred.predict.run_predictions_for_event') as mock_predict:
            mock_predict.return_value = fake_results
            jc = MagicMock()

            manager._predict_round(jc, 2024, 1, {"raceName": "Normal GP", "round": 1})
