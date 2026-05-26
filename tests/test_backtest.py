import pytest
from unittest.mock import MagicMock, patch
from f1pred.backtest import _auto_backtest_seasons, run_backtests

def test_auto_backtest_seasons_success():
    jc = MagicMock()
    jc.get_season_schedule.return_value = [{"season": "2024"}]
    years = _auto_backtest_seasons(jc)
    assert years == [2019, 2020, 2021, 2022, 2023]

def test_auto_backtest_seasons_fallback():
    jc = MagicMock()
    jc.get_season_schedule.side_effect = Exception("API down")
    years = _auto_backtest_seasons(jc)
    assert years == [2020, 2021, 2022, 2023]

@patch("f1pred.backtest.JolpicaClient")
@patch("f1pred.backtest.run_predictions_for_event")
@patch("f1pred.backtest.compute_event_metrics")
def test_run_backtests_auto(mock_metrics, mock_predict, mock_jc_class):
    mock_jc = mock_jc_class.return_value
    mock_jc.get_season_schedule.side_effect = lambda season: {
        "current": [{"season": "2024"}],
        "2019": [{"round": "1", "raceName": "Race 1"}],
        "2020": [],
        "2021": [],
        "2022": [],
        "2023": []
    }[season]

    mock_predict.return_value = {
        "season": "2019",
        "round": "1",
        "sessions": {
            "race": {
                "ranked": [],
                "prob_matrix": [],
                "pairwise": []
            }
        }
    }

    mock_metrics.return_value = {"metric": 1}

    cfg = MagicMock()
    cfg.data_sources.jolpica.base_url = "http://fake"
    cfg.data_sources.jolpica.timeout_seconds = 10
    cfg.data_sources.jolpica.rate_limit_sleep = 0
    cfg.backtesting.seasons = "auto"
    cfg.modelling.targets.session_types = ["race"]

    run_backtests(cfg)

    mock_predict.assert_called_once()
    mock_metrics.assert_called_once()


@patch("f1pred.backtest.JolpicaClient")
@patch("f1pred.backtest.run_predictions_for_event")
def test_run_backtests_all_seasons(mock_predict, mock_jc_class):
    mock_jc = mock_jc_class.return_value
    mock_jc.get_season_schedule.return_value = []

    cfg = MagicMock()
    cfg.backtesting.seasons = "all"

    run_backtests(cfg)
    # Just checking it doesn't crash, schedule loop runs over all years

@patch("f1pred.backtest.JolpicaClient")
@patch("f1pred.backtest.run_predictions_for_event")
def test_run_backtests_list_seasons(mock_predict, mock_jc_class):
    mock_jc = mock_jc_class.return_value
    mock_jc.get_season_schedule.side_effect = lambda s: [{"round": "1", "raceName": "Race"} if s == "2023" else []]

    mock_predict.return_value = None # None means continue

    cfg = MagicMock()
    cfg.backtesting.seasons = [2023]

    run_backtests(cfg)
    mock_predict.assert_called_once()

@patch("f1pred.backtest.JolpicaClient")
@patch("f1pred.backtest.run_predictions_for_event")
def test_run_backtests_exception_handled(mock_predict, mock_jc_class):
    mock_jc = mock_jc_class.return_value
    mock_jc.get_season_schedule.return_value = [{"round": "1", "raceName": "Race"}]

    mock_predict.side_effect = Exception("Prediction failed")

    cfg = MagicMock()
    cfg.backtesting.seasons = [2023]

    run_backtests(cfg)
    mock_predict.assert_called_once() # Exception is caught and logged
