import pytest
import pandas as pd
from unittest.mock import MagicMock, patch
from f1pred.prediction_manager import PredictionManager, _fingerprint_predictions

@pytest.fixture
def cfg():
    cfg = MagicMock()
    cfg.app.webhook_min_stable_cycles = 1
    cfg.modelling.targets.session_types = ["race"]
    cfg.paths.cache_dir = "cache"
    return cfg

def make_preds(positions):
    data = []
    for d_id, pos in positions.items():
        data.append({
            "driverId": d_id,
            "predicted_position": pos,
            "code": d_id.upper()[:3],
            "name": d_id,
            "constructorName": "Team",
            "p_win": 0.1,
            "p_top3": 0.3,
            "mean_pos": float(pos),
        })
    return pd.DataFrame(data)

def test_immediate_webhook_when_cycles_0(cfg):
    cfg.app.webhook_min_stable_cycles = 0
    manager = PredictionManager(cfg)

    # First run establishes baseline (no diff to send yet, but sets last_sent)
    preds1 = make_preds({"ver": 1, "ham": 2})
    results1 = {"season": "2024", "round": "1", "sessions": {"race": {"ranked": preds1, "meta": {}}}}

    with patch("f1pred.predict.run_predictions_for_event", return_value=results1):
        manager._predict_round(MagicMock(), 2024, 1, {"raceName": "GP"})

    assert "1_race" in manager._last_sent_webhook_fps

    # Second run with change
    preds2 = make_preds({"ver": 2, "ham": 1})
    results2 = {"season": "2024", "round": "1", "sessions": {"race": {"ranked": preds2, "meta": {}}}}

    with patch("f1pred.predict.run_predictions_for_event", return_value=results2):
        with patch.object(manager, "_send_discord_webhook") as mock_webhook:
            manager._predict_round(MagicMock(), 2024, 1, {"raceName": "GP"})
            mock_webhook.assert_called_once()

def test_debounced_webhook_stable(cfg):
    cfg.app.webhook_min_stable_cycles = 1
    manager = PredictionManager(cfg)

    # Establish baseline
    preds1 = make_preds({"ver": 1, "ham": 2})
    results1 = {"season": "2024", "round": "1", "sessions": {"race": {"ranked": preds1, "meta": {}}}}
    with patch("f1pred.predict.run_predictions_for_event", return_value=results1):
        manager._predict_round(MagicMock(), 2024, 1, {"raceName": "GP"})

    # Change happens
    preds2 = make_preds({"ver": 2, "ham": 1})
    results2 = {"season": "2024", "round": "1", "sessions": {"race": {"ranked": preds2, "meta": {}}}}

    with patch("f1pred.predict.run_predictions_for_event", return_value=results2):
        with patch.object(manager, "_send_discord_webhook") as mock_webhook:
            # First call: mark as pending (stable cycles = 0)
            manager._predict_round(MagicMock(), 2024, 1, {"raceName": "GP"})
            mock_webhook.assert_not_called()
            assert "1_race" in manager._pending_webhook_fps
            assert manager._pending_webhook_cycles["1_race"] == 0

            # Second call: stable (stable cycles = 1)
            manager._predict_round(MagicMock(), 2024, 1, {"raceName": "GP"})
            mock_webhook.assert_called_once()
            assert "1_race" not in manager._pending_webhook_fps

def test_webhook_no_send_if_flapping(cfg):
    cfg.app.webhook_min_stable_cycles = 1
    manager = PredictionManager(cfg)

    # Baseline: A
    predsA = make_preds({"ver": 1, "ham": 2})
    resultsA = {"season": "2024", "round": "1", "sessions": {"race": {"ranked": predsA, "meta": {}}}}
    with patch("f1pred.predict.run_predictions_for_event", return_value=resultsA):
        manager._predict_round(MagicMock(), 2024, 1, {"raceName": "GP"})

    # Change to B
    predsB = make_preds({"ver": 2, "ham": 1})
    resultsB = {"season": "2024", "round": "1", "sessions": {"race": {"ranked": predsB, "meta": {}}}}
    with patch("f1pred.predict.run_predictions_for_event", return_value=resultsB):
        manager._predict_round(MagicMock(), 2024, 1, {"raceName": "GP"})
        assert "1_race" in manager._pending_webhook_fps

    # Change back to A before debounce expires
    with patch("f1pred.predict.run_predictions_for_event", return_value=resultsA):
        with patch.object(manager, "_send_discord_webhook") as mock_webhook:
            manager._predict_round(MagicMock(), 2024, 1, {"raceName": "GP"})
            mock_webhook.assert_not_called()
            assert "1_race" not in manager._pending_webhook_fps # Pending cleared because it matches last sent

def test_diff_calculated_against_last_sent(cfg):
    cfg.app.webhook_min_stable_cycles = 1
    manager = PredictionManager(cfg)

    # 1. Baseline: ver=1, ham=2
    preds1 = make_preds({"ver": 1, "ham": 2})
    results1 = {"season": "2024", "round": "1", "sessions": {"race": {"ranked": preds1, "meta": {}}}}
    with patch("f1pred.predict.run_predictions_for_event", return_value=results1):
        manager._predict_round(MagicMock(), 2024, 1, {"raceName": "GP"})

    # 2. Update 1 (Debounced/SSE only): ver=2, ham=1
    preds2 = make_preds({"ver": 2, "ham": 1})
    results2 = {"season": "2024", "round": "1", "sessions": {"race": {"ranked": preds2, "meta": {}}}}
    with patch("f1pred.predict.run_predictions_for_event", return_value=results2):
        # mark as pending
        manager._predict_round(MagicMock(), 2024, 1, {"raceName": "GP"})

    # 3. Update 2: ver=3, ham=1, rus=2
    preds3 = make_preds({"ver": 3, "ham": 1, "rus": 2})
    results3 = {"season": "2024", "round": "1", "sessions": {"race": {"ranked": preds3, "meta": {}}}}
    with patch("f1pred.predict.run_predictions_for_event", return_value=results3):
        # First call to Update 2: mark as pending (re-timer)
        manager._predict_round(MagicMock(), 2024, 1, {"raceName": "GP"})

        # Second call to Update 2: Stable, trigger
        with patch.object(manager, "_send_discord_webhook") as mock_webhook:
            manager._predict_round(MagicMock(), 2024, 1, {"raceName": "GP"})

            mock_webhook.assert_called_once()
            # The diff should be against results1 (ver: 1->3), not results2 (ver: 2->3)
            args, _ = mock_webhook.call_args
            updates = args[1]
            diff = updates["race"][0]

            ver_move = next(m for m in diff.movements if m.driver_id == "ver")
            assert ver_move.old_position == 1
            assert ver_move.new_position == 3
