## 2024-05-11 - "Settings" Navigation Tab Role
**Learning:** The navigation tabs in `index.html` (e.g., "Predictions" and "Settings" buttons around line 180) currently lack ARIA `role="tab"` attributes. When writing automated Playwright tests, `page.get_by_role("tab", name="Settings")` will fail to find the button.
**Action:** When testing these tabs, use text-based locators like `page.locator("button:has-text('Settings')")` instead, or propose a separate UX improvement to add proper ARIA tablist/tab roles to this navigation structure.

## 2025-02-24 - Form validation required indicators
**Learning:** A common UX/A11y pattern is explicitly marking `required` form fields. While `required` attributes ensure client-side validation logic catches empty fields, without a visual indicator (like a red asterisk `*`) users may only discover the field is required *after* a failed submission.
**Action:** Always pair `required` inputs with explicit visual indicators (like `<span class="text-red-500">*</span>`) appended to the associated `<label>` element, and provide helpful tooltips on disabled actionable elements.
## 2026-06-08 - Accessible Tooltips for Disabled States
**Learning:** `pointer-events-none` on parent wrappers completely blocks pointer events, making it impossible for users to interact with or even see tooltips (like `title` attributes) on disabled elements nested inside.
**Action:** Remove `pointer-events-none` from parent wrappers and apply `disabled:cursor-not-allowed` and dynamic `title` attributes directly to the interactive elements (buttons/inputs) themselves so screen readers and mouse users get context on why the element is disabled.
