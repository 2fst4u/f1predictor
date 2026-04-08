import pytest
from unittest.mock import patch, MagicMock

from f1pred.live import live_loop

@patch("f1pred.live.JolpicaClient")
@patch("f1pred.live.resolve_event")
@patch("f1pred.live.run_predictions_for_event")
@patch("f1pred.live.print_countdown")
def test_live_loop(mock_print_countdown, mock_run_predictions, mock_resolve_event, mock_jolpica_client):
    # Arrange
    cfg = MagicMock()
    cfg.data_sources.jolpica.base_url = "http://fake-jolpica-url"
    cfg.data_sources.jolpica.timeout_seconds = 10
    cfg.data_sources.jolpica.rate_limit_sleep = 1.0
    cfg.app.live_refresh_seconds = 30

    season = "2024"
    rnd = "5"
    sessions = ["race"]

    mock_resolve_event.return_value = (2024, 5, {"raceName": "Miami Grand Prix"})

    # We want the loop to run once and then exit to prevent an infinite loop in the test
    mock_print_countdown.side_effect = StopIteration("Break infinite loop")

    # Act
    with pytest.raises(StopIteration, match="Break infinite loop"):
        live_loop(cfg, season, rnd, sessions)

    # Assert
    mock_jolpica_client.assert_called_once_with("http://fake-jolpica-url", 10, 1.0)
    mock_resolve_event.assert_called_once_with(mock_jolpica_client.return_value, season, rnd)
    mock_run_predictions.assert_called_once_with(
        cfg,
        season="2024",
        rnd="5",
        sessions=["race"]
    )
    mock_print_countdown.assert_called_once_with(30, "Next update in")
