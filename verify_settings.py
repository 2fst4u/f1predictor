import os
from playwright.sync_api import sync_playwright

def run_cuj(page):
    # Navigate to app
    page.goto("http://localhost:8000")
    page.wait_for_timeout(1000)

    # Click Settings tab
    page.get_by_role("button", name="Settings").click()
    page.wait_for_timeout(500)

    # Take screenshot of login form
    page.screenshot(path="/home/jules/verification/screenshots/login.png")

    # Fill in login form
    page.locator("input[type='text']").fill("admin")
    page.wait_for_timeout(500)
    page.locator("input[type='password']").first.fill("admin")
    page.wait_for_timeout(500)
    page.locator("button:has-text('Login')").click()
    page.wait_for_timeout(3000)

    # Take screenshot of settings form
    page.screenshot(path="/home/jules/verification/screenshots/settings.png")

    # Fill out discord webhook
    page.locator("input[type='text']").last.fill("https://discord.com/api/webhooks/123/abc")
    page.wait_for_timeout(500)
    page.evaluate("document.querySelector('button[type=\"submit\"]').click()")
    page.wait_for_timeout(1000)

    # Take screenshot of success state
    page.screenshot(path="/home/jules/verification/screenshots/settings_saved.png")

    # Switch back to Predictions tab
    page.get_by_role("button", name="Predictions").click()
    page.wait_for_timeout(1000)

if __name__ == "__main__":
    os.makedirs("/home/jules/verification/screenshots", exist_ok=True)
    os.makedirs("/home/jules/verification/videos", exist_ok=True)
    with sync_playwright() as p:
        browser = p.chromium.launch(headless=True)
        context = browser.new_context(
            record_video_dir="/home/jules/verification/videos"
        )
        page = context.new_page()
        try:
            run_cuj(page)
        finally:
            context.close()
            browser.close()