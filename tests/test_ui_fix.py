import os
import json
import pytest

try:
    from playwright.sync_api import Page, expect
    HAS_PLAYWRIGHT = True
except ImportError:
    HAS_PLAYWRIGHT = False

@pytest.mark.skipif(not HAS_PLAYWRIGHT or os.environ.get("CI") == "true", reason="Skipping Playwright tests in CI environment or if playwright is missing")
def test_ui_logic_v3(page: "Page"):
    """Verify UI logic v3: immediate load, caching and session updates."""
    abs_path = os.path.abspath("f1pred/templates/index.html")
    page.goto(f"file://{abs_path}")

    # Wait for Alpine.js
    page.wait_for_function("window.Alpine !== undefined")

    # 1. Mock runPrediction to detect calls
    page.evaluate("""
        const el = document.querySelector('[x-data]');
        const data = Alpine.$data(el);
        data._runPredictionCalled = false;
        data.runPrediction = async function() {
            this._runPredictionCalled = true;
            // Simulate results being returned and cached
            this.results = {
                season: '2024',
                round: this.params.round,
                sessions: { race: { predictions: [] } }
            };
            this.allRoundsData[String(this.results.round)] = this.results;
        };
        // Mock fetchEventStatus
        data.fetchEventStatus = async function() { return; };
        // Mock fetchSchedule
        data.fetchSchedule = async function() { return; };
    """)

    # 2. Test Initial Load Prediction
    # Mock config with next_event
    page.evaluate("""
        const el = document.querySelector('[x-data]');
        const data = Alpine.$data(el);
        data.config = { next_event: { season: '2024', round: '1' } };
        // Force init logic trigger for next_event if we weren't in init already
        // Actually, let's just manually call the logic we added to init
    """)

    # We call selectRound as if init() just did it
    page.evaluate("Alpine.$data(document.querySelector('[x-data]')).selectRound('1')")

    # Verify runPrediction was called
    assert page.evaluate("Alpine.$data(document.querySelector('[x-data]'))._runPredictionCalled") is True

    # Reset flag
    page.evaluate("Alpine.$data(document.querySelector('[x-data]'))._runPredictionCalled = false")

    # 3. Test caching with mixed types (int vs string)
    # Inject round 2 as int in cache
    page.evaluate("""
        const el = document.querySelector('[x-data]');
        const data = Alpine.$data(el);
        data.allRoundsData['2'] = { round: 2, sessions: { qualifying: {} } };
    """)

    # Select round 2
    page.evaluate("Alpine.$data(document.querySelector('[x-data]')).selectRound(2)")

    # Verify results are set from cache and runPrediction NOT called
    results_round = page.evaluate("Alpine.$data(document.querySelector('[x-data]')).results.round")
    assert results_round == 2
    assert page.evaluate("Alpine.$data(document.querySelector('[x-data]'))._runPredictionCalled") is False

    # Verify activeSession was updated to qualifying
    active_session = page.evaluate("Alpine.$data(document.querySelector('[x-data]')).activeSession")
    assert active_session == 'qualifying'
