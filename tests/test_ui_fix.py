import os
import json
import pytest

try:
    from playwright.sync_api import Page, expect
    HAS_PLAYWRIGHT = True
except ImportError:
    HAS_PLAYWRIGHT = False

@pytest.mark.skipif(not HAS_PLAYWRIGHT or os.environ.get("CI") == "true", reason="Skipping Playwright tests in CI environment or if playwright is missing")
def test_ui_caching(page: "Page"):
    """Verify UI caching logic in index.html."""
    abs_path = os.path.abspath("f1pred/templates/index.html")
    page.goto(f"file://{abs_path}")

    # Wait for Alpine.js
    page.wait_for_function("window.Alpine !== undefined")

    # 1. Inject Mock Data for Round 1
    mock_results_r1 = {
        "season": "2024",
        "round": "1",
        "sessions": {
            "race": {
                "predictions": [{"driverId": "verstappen", "predicted_position": 1}],
                "weather": {}
            }
        }
    }

    page.evaluate(f"""
        const el = document.querySelector('[x-data]');
        const data = Alpine.$data(el);
        data.allRoundsData['1'] = {json.dumps(mock_results_r1)};
        data.params.round = '1';
        data.results = data.allRoundsData['1'];
        data.activeSession = 'race';
    """)

    # 2. Mock runPrediction to detect calls
    page.evaluate("""
        const el = document.querySelector('[x-data]');
        const data = Alpine.$data(el);
        data._runPredictionCalled = false;
        const originalRunPrediction = data.runPrediction;
        data.runPrediction = async function() {
            this._runPredictionCalled = true;
            return; // Don't actually run it
        };
    """)

    # 3. Mock fetchEventStatus to avoid actual network call
    page.evaluate("""
        const el = document.querySelector('[x-data]');
        const data = Alpine.$data(el);
        data.fetchEventStatus = async function() { return; };
    """)

    # 4. Trigger selectRound for round 2 (UNC ACHED)
    page.evaluate("Alpine.$data(document.querySelector('[x-data]')).selectRound('2')")

    # Verify runPrediction WAS called for Round 2
    was_called_r2 = page.evaluate("Alpine.$data(document.querySelector('[x-data]'))._runPredictionCalled")
    assert was_called_r2 is True, "runPrediction should be called for uncached round 2"

    # Reset the flag
    page.evaluate("Alpine.$data(document.querySelector('[x-data]'))._runPredictionCalled = false")

    # 5. Trigger selectRound for round 1 (CACHED)
    page.evaluate("Alpine.$data(document.querySelector('[x-data]')).selectRound('1')")

    # Verify runPrediction WAS NOT called for Round 1
    was_called_r1 = page.evaluate("Alpine.$data(document.querySelector('[x-data]'))._runPredictionCalled")
    assert was_called_r1 is False, "runPrediction should NOT be called for cached round 1"
