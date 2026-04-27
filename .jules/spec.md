
### Mocking local imports
When attempting to mock functions that are locally imported within a method's body (like `from .predict import run_predictions_for_event` inside `PredictionManager._run_prediction_cycle()`), attempting to patch the calling module's namespace (e.g. `f1pred.prediction_manager.run_predictions_for_event`) will fail with an AttributeError because the function doesn't exist on the module level. You must patch the original module where the function resides (e.g., `@patch('f1pred.predict.run_predictions_for_event')`).

### Avoiding application logic changes for tests
When an application's data transformations (e.g., rounding) crash on `None` values (like `math.isnan`), do not alter the application code to handle the `None` just so a test can assert how it behaves. Instead, write tests that pass valid types or verify the exception is handled upstream if that is the intended behavior.
## 2024-05-18 - PredictionCache Mocking and Coverage
When adding tests to `PredictionCache` (in `f1pred/util.py`) for the explicit `get_by_key` and `set_by_key` methods, you must mock `hashlib.sha256` to force a write error for testing error handling since standard OS/file permissions can be tricky to guarantee a write failure across all environments safely.
