import os
import json
import pytest

try:
    from playwright.sync_api import Page, expect
    HAS_PLAYWRIGHT = True
except ImportError:
    HAS_PLAYWRIGHT = False

@pytest.mark.skipif(not HAS_PLAYWRIGHT or os.environ.get("CI") == "true", reason="Skipping Playwright tests in CI environment or if playwright is missing")
def test_ui_logic_v6_sync(page: "Page"):
    """Verify UI logic v6: Background sync doesn't overwrite wrong round."""
    abs_path = os.path.abspath("f1pred/templates/index.html")

    page.goto(f"file://{abs_path}")

    # Wait for Alpine.js
    page.wait_for_function("window.Alpine !== undefined", timeout=10000)

    # 1. Setup state: Looking at Round 4 (which has no results yet)
    page.evaluate("""
        const el = document.querySelector('[x-data]');
        const data = Alpine.$data(el);
        data.params.season = '2024';
        data.params.round = '4';
        data.results = null;
        data.allRoundsData = {};
    """)

    # 2. Simulate arrival of Round 2 background update
    page.evaluate("""
        const el = document.querySelector('[x-data]');
        const data = Alpine.$data(el);
        const payload = {
            type: 'prediction_round',
            data: {
                season: '2024',
                round: '2',
                sessions: { race: { predictions: [] } }
            }
        };
        data.handleLiveEvent(payload);
    """)

    # 3. Verify results are still null (Round 2 shouldn't overwrite active Round 4)
    current_results = page.evaluate("Alpine.$data(document.querySelector('[x-data]')).results")
    assert current_results is None

    # 4. Verify Round 2 IS in cache
    has_r2_in_cache = page.evaluate("Alpine.$data(document.querySelector('[x-data]')).allRoundsData['2'] !== undefined")
    assert has_r2_in_cache is True

    # 5. Simulate arrival of Round 4 background update
    page.evaluate("""
        const el = document.querySelector('[x-data]');
        const data = Alpine.$data(el);
        const payload = {
            type: 'prediction_round',
            data: {
                season: '2024',
                round: '4',
                sessions: { race: { predictions: [] } }
            }
        };
        data.handleLiveEvent(payload);
    """)

    # 6. Verify results are NOW set (Round 4 matches active)
    active_results_round = page.evaluate("Alpine.$data(document.querySelector('[x-data]')).results.round")
    assert str(active_results_round) == '4'
