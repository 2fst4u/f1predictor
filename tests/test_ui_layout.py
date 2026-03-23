import os
import json
import pytest
from playwright.sync_api import Page, expect

@pytest.mark.skipif(os.environ.get("CI") == "true", reason="Skipping Playwright tests in CI environment")
def test_model_mix_width_and_offsets(page: Page):
    """Verify Model Mix graph width and factor limit offsets in the UI."""
    abs_path = os.path.abspath("f1pred/templates/index.html")
    page.goto(f"file://{abs_path}")

    # Wait for Alpine to initialize
    page.wait_for_function("window.Alpine !== undefined")

    # Mock data injection
    mock_results = {
        "season": "2024",
        "round": "1",
        "sessions": {
            "race": {
                "predictions": [
                    {
                        "driverId": "verstappen",
                        "name": "Max Verstappen",
                        "code": "VER",
                        "constructorName": "Red Bull",
                        "predicted_position": 1,
                        "p_win": 0.8,
                        "p_top3": 0.95,
                        "p_dnf": 0.05,
                        "ensemble_components": {
                            "gbm": 0.4,
                            "elo": 0.2,
                            "bt": 0.2,
                            "mixed": 0.2
                        },
                        "shap_values": {
                            "temp_skill": -0.5,
                            "rain_skill": -0.3,
                            "circuit_experience": -0.2
                        }
                    }
                ],
                "weather": {"temp_mean": 25, "rain_sum": 0, "wind_mean": 10}
            }
        }
    }

    page.evaluate(f"""
        const el = document.querySelector('[x-data]');
        const data = Alpine.$data(el);
        data.results = {json.dumps(mock_results)};
        data.activeSession = 'race';
    """)

    # 1. Verify CSS Classes for Width
    model_mix_container = page.locator(".md\\:w-64").first
    expect(model_mix_container).to_be_visible()

    progress_bar = page.locator(".sm\\:w-64").first
    expect(progress_bar).to_be_visible()

    # 2. Verify JS getFactorLimit Offsets
    # Test Desktop (Large Viewport)
    page.set_viewport_size({"width": 1200, "height": 800})
    page.evaluate("Alpine.$data(document.querySelector('[x-data]')).windowWidth = 1200")
    limit_large = page.evaluate("Alpine.$data(document.querySelector('[x-data]')).getFactorLimit()")
    # Expected for 1200px: w >= 1024 -> offset = 380. containerW = 1200.
    # available = 1200 - 380 = 820. count = floor(820 / 156) = 5.
    assert limit_large == 5

    # Test Desktop (Medium Viewport)
    page.set_viewport_size({"width": 800, "height": 800})
    page.evaluate("Alpine.$data(document.querySelector('[x-data]')).windowWidth = 800")
    limit_medium = page.evaluate("Alpine.$data(document.querySelector('[x-data]')).getFactorLimit()")
    # Expected for 800px: w < 1024 -> offset = 350. containerW = 800.
    # available = 800 - 350 = 450. count = floor(450 / 156) = 2.
    assert limit_medium == 2

    # Test Mobile
    page.set_viewport_size({"width": 400, "height": 800})
    page.evaluate("Alpine.$data(document.querySelector('[x-data]')).windowWidth = 400")
    limit_mobile = page.evaluate("Alpine.$data(document.querySelector('[x-data]')).getFactorLimit()")
    # Expected for 400px: w < 768 -> floor((400 - 32) / 119) = floor(368 / 119) = 3.
    assert limit_mobile == 3
