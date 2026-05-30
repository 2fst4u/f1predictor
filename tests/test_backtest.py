import pytest
from unittest.mock import MagicMock, patch
from f1pred.backtest import (
    _auto_backtest_seasons,
    run_backtests,
    summarize_metrics,
    compare_summaries,
)


def test_summarize_metrics_means_and_grouping():
    rows = [
        {"event": "race", "spearman": 0.8, "accuracy_top3": 1.0, "crps": 0.2},
        {"event": "race", "spearman": 0.6, "accuracy_top3": 0.0, "crps": 0.4},
        {"event": "qualifying", "spearman": 0.9, "accuracy_top3": 1.0, "crps": 0.1},
    ]
    s = summarize_metrics(rows)
    assert s["n_sessions"] == 3
    assert s["overall"]["spearman"]["mean"] == pytest.approx(0.7666667, abs=1e-4)
    assert s["overall"]["spearman"]["n"] == 3
    # Per-session breakout is preserved.
    assert s["by_session"]["qualifying"]["spearman"]["mean"] == pytest.approx(0.9)
    assert s["by_session"]["race"]["accuracy_top3"]["mean"] == pytest.approx(0.5)


def test_summarize_metrics_skips_nan_and_missing():
    rows = [
        {"event": "race", "spearman": float("nan"), "crps": 0.2},
        {"event": "race", "spearman": 0.5},  # no crps key
    ]
    s = summarize_metrics(rows)
    assert s["overall"]["spearman"]["n"] == 1  # NaN skipped
    assert s["overall"]["spearman"]["mean"] == pytest.approx(0.5)
    assert s["overall"]["crps"]["n"] == 1


def test_compare_summaries_respects_direction():
    base = summarize_metrics([{"event": "race", "spearman": 0.6, "crps": 0.4}])
    cand = summarize_metrics([{"event": "race", "spearman": 0.7, "crps": 0.3}])
    cmp = compare_summaries(base, cand)
    # spearman is higher-is-better: +0.1 is an improvement.
    assert cmp["spearman"]["improved"] is True
    assert cmp["spearman"]["delta"] == pytest.approx(0.1)
    # crps is lower-is-better: -0.1 is an improvement.
    assert cmp["crps"]["improved"] is True
    assert cmp["crps"]["delta"] == pytest.approx(-0.1)


def test_compare_summaries_detects_regression():
    base = summarize_metrics([{"event": "race", "spearman": 0.8}])
    cand = summarize_metrics([{"event": "race", "spearman": 0.5}])
    cmp = compare_summaries(base, cand)
    assert cmp["spearman"]["improved"] is False

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
