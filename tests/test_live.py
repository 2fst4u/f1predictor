import pytest
from unittest.mock import patch, MagicMock
from f1pred.live import live_loop

def test_live_loop_executes_and_exits():
    # Arrange
    cfg = MagicMock()
    cfg.data_sources.jolpica.base_url = "http://test"
    cfg.data_sources.jolpica.timeout_seconds = 10
    cfg.data_sources.jolpica.rate_limit_sleep = 0
    cfg.app.live_refresh_seconds = 60

    with patch('f1pred.live.JolpicaClient') as mock_jc_cls, \
         patch('f1pred.live.resolve_event') as mock_resolve, \
         patch('f1pred.live.run_predictions_for_event') as mock_run_preds, \
         patch('f1pred.live.print_countdown', side_effect=StopIteration) as mock_countdown:

        mock_resolve.return_value = (2024, 1, {'raceName': 'Test GP'})

        # Act
        with pytest.raises(StopIteration):
            live_loop(cfg, season="2024", rnd="1", sessions=["race"])

        # Assert
        mock_resolve.assert_called_once()
        mock_run_preds.assert_called_once_with(
            cfg,
            season="2024",
            rnd="1",
            sessions=["race"]
        )
        mock_countdown.assert_called_once_with(60, "Next update in")
