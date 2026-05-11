## 2024-05-11 - "Settings" Navigation Tab Role
**Learning:** The navigation tabs in `index.html` (e.g., "Predictions" and "Settings" buttons around line 180) currently lack ARIA `role="tab"` attributes. When writing automated Playwright tests, `page.get_by_role("tab", name="Settings")` will fail to find the button.
**Action:** When testing these tabs, use text-based locators like `page.locator("button:has-text('Settings')")` instead, or propose a separate UX improvement to add proper ARIA tablist/tab roles to this navigation structure.
