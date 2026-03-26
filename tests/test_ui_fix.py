import os
import json
import pytest

try:
    from playwright.sync_api import Page, expect
    HAS_PLAYWRIGHT = True
except ImportError:
    HAS_PLAYWRIGHT = False

@pytest.mark.skipif(not HAS_PLAYWRIGHT or os.environ.get("CI") == "true", reason="Skipping Playwright tests in CI environment or if playwright is missing")
def test_ui_logic(page: "Page"):
    """Verify core UI logic in index.html."""
    abs_path = os.path.abspath("f1pred/templates/index.html")
    page.goto(f"file://{abs_path}")

    # Wait for Alpine to initialize
    page.wait_for_function("window.Alpine !== undefined")

    # Mock data injection with multiple sessions
    mock_results = {
        "season": "2024",
        "round": "1",
        "sessions": {
            "qualifying": {
                "predictions": [{"driverId": "verstappen", "predicted_position": 1, "p_win": 0.5, "p_top3": 0.8}],
                "weather": {}
            },
            "race": {
                "predictions": [{"driverId": "verstappen", "predicted_position": 1, "p_win": 0.6, "p_top3": 0.9}],
                "weather": {}
            }
        }
    }

    page.evaluate(f"""
        const el = document.querySelector('[x-data]');
        const data = Alpine.$data(el);
        data.results = {json.dumps(mock_results)};
    """)

    # 1. Verify getAvailableSessions()
    available_sessions = page.evaluate("Alpine.$data(document.querySelector('[x-data]')).getAvailableSessions()")
    # Order: ['qualifying', 'race']
    assert available_sessions == ['qualifying', 'race']

    # 2. Verify Qualifying is visible in tabs
    page.wait_for_selector("#tab-qualifying")
    expect(page.locator("#tab-qualifying")).to_be_visible()
    expect(page.locator("#tab-race")).to_be_visible()

    # 3. Verify Empty State text
    # Set results to null to trigger empty state
    page.evaluate("Alpine.$data(document.querySelector('[x-data]')).results = null")
    empty_state_text = page.locator("p.text-xl.font-light").text_content()
    assert "Select a race to view predictions" in empty_state_text
