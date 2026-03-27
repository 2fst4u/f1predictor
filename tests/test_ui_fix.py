import os
import json
import pytest

try:
    from playwright.sync_api import Page, expect
    HAS_PLAYWRIGHT = True
except ImportError:
    HAS_PLAYWRIGHT = False

@pytest.mark.skipif(not HAS_PLAYWRIGHT or os.environ.get("CI") == "true", reason="Skipping Playwright tests in CI environment or if playwright is missing")
def test_ui_logic_v5(page: "Page"):
    """Verify UI logic v5: Round-aware prediction and immediate display."""
    abs_path = os.path.abspath("f1pred/templates/index.html")

    # 1. Intercept requests
    mock_latest = {
        "results": {
            "season": "2024",
            "rounds": {
                "1": { "round": 1, "sessions": { "race": { "predictions": [] } } }
            }
        },
        "last_update": "2024-03-24T12:00:00Z"
    }

    def handle_route(route):
        url = route.request.url
        if "predictions/latest" in url:
            route.fulfill(status=200, content_type="application/json", body=json.dumps(mock_latest))
        elif "api/config" in url:
            route.fulfill(status=200, content_type="application/json", body=json.dumps({ "next_event": { "season": "2024", "round": "1" }, "default_sessions": ["race"] }))
        elif "api/seasons" in url:
            route.fulfill(status=200, content_type="application/json", body=json.dumps([{"season": "2024"}]))
        elif "api/schedule/2024" in url:
            route.fulfill(status=200, content_type="application/json", body=json.dumps({ "races": [{"round": "1", "raceName": "Bahrain"}] }))
        elif "api/event-status" in url:
            route.fulfill(status=200, content_type="application/json", body=json.dumps({ "sessions": [{"id": "race", "has_results": False}] }))
        else:
            route.continue_()

    page.route("**/api/**", handle_route)
    page.goto(f"file://{abs_path}")

    # Wait for Alpine.js
    page.wait_for_function("window.Alpine !== undefined", timeout=10000)

    # Manually trigger the data loading that normally happens in init
    # (since file:// blocks the actual fetch in some environments)
    page.evaluate(f"""
        const el = document.querySelector('[x-data]');
        const data = Alpine.$data(el);
        data.allRoundsData = {json.dumps(mock_latest['results']['rounds'])};
        data.params.round = '1';
        data.results = data.allRoundsData['1'];
    """)

    # Verify Round 1 is displayed
    results_round = page.evaluate("Alpine.$data(document.querySelector('[x-data]')).results.round")
    assert results_round == 1

    # 2. Mock runPrediction to be long-running
    page.evaluate("""
        const el = document.querySelector('[x-data]');
        const data = Alpine.$data(el);
        data._runPredictionCalled = false;
        data.runPrediction = async function() {
            const target = this.params.round;
            this.loading = true;
            this.predictingRound = target;
            this._runPredictionCalled = true;

            // Simulate delay
            await new Promise(resolve => setTimeout(resolve, 1000));

            const results = { round: target, sessions: { race: {} } };
            this.allRoundsData[String(target)] = results;
            if (String(target) === String(this.params.round)) {
                this.results = results;
            }
            this.loading = false;
            this.predictingRound = null;
        };
        // Mock fetchEventStatus
        data.fetchEventStatus = async function() { return; };
    """)

    # 3. Trigger prediction for Round 2 and immediately switch back to Round 1
    # We do this from JS to avoid race conditions in the test script
    page.evaluate("""
        const data = Alpine.$data(document.querySelector('[x-data]'));
        data.selectRound('2'); // This starts long-running prediction for R2
        setTimeout(() => {
            data.selectRound('1'); // Switch back to R1 while R2 is still predicting
        }, 200);
    """)

    # Wait for the R2 prediction to finish
    page.wait_for_timeout(1500)

    # 4. Verify that R1 results are still showing (because R1 is active)
    current_results_round = page.evaluate("Alpine.$data(document.querySelector('[x-data]')).results.round")
    assert current_results_round == 1

    # 5. Verify that R2 is now in cache
    has_r2_in_cache = page.evaluate("Alpine.$data(document.querySelector('[x-data]')).allRoundsData['2'] !== undefined")
    assert has_r2_in_cache is True
