## 2024-05-11 - "Settings" Navigation Tab Role
**Learning:** The navigation tabs in `index.html` (e.g., "Predictions" and "Settings" buttons around line 180) currently lack ARIA `role="tab"` attributes. When writing automated Playwright tests, `page.get_by_role("tab", name="Settings")` will fail to find the button.
**Action:** When testing these tabs, use text-based locators like `page.locator("button:has-text('Settings')")` instead, or propose a separate UX improvement to add proper ARIA tablist/tab roles to this navigation structure.

## 2025-02-24 - Form validation required indicators
**Learning:** A common UX/A11y pattern is explicitly marking `required` form fields. While `required` attributes ensure client-side validation logic catches empty fields, without a visual indicator (like a red asterisk `*`) users may only discover the field is required *after* a failed submission.
**Action:** Always pair `required` inputs with explicit visual indicators (like `<span class="text-red-500">*</span>`) appended to the associated `<label>` element, and provide helpful tooltips on disabled actionable elements.
## 2026-06-08 - Accessible Tooltips for Disabled States
**Learning:** `pointer-events-none` on parent wrappers completely blocks pointer events, making it impossible for users to interact with or even see tooltips (like `title` attributes) on disabled elements nested inside.
**Action:** Remove `pointer-events-none` from parent wrappers and apply `disabled:cursor-not-allowed` and dynamic `title` attributes directly to the interactive elements (buttons/inputs) themselves so screen readers and mouse users get context on why the element is disabled.
## 2024-06-15 - Disabled States Accessibility
**Learning:** Applying `pointer-events-none` to a parent container blocks pointer events on all children. This prevents disabled child elements from showing `cursor-not-allowed` and makes `title` tooltips inaccessible on hover, ruining the UX for users trying to understand why an element is disabled. Also, applying opacity to a parent compounds with child opacity, leading to poor contrast.
**Action:** Remove `pointer-events-none` and `opacity-50` from parent wrappers of disabled groups. Apply `disabled:opacity-50`, `disabled:cursor-not-allowed`, and informative `title` attributes directly to the interactive child elements to preserve tooltip accessibility and clear interaction feedback.
## 2024-05-18 - Disabled Input Tooltips and Opacity
**Learning:** When a parent wrapper has `opacity-50`, it compounds with the opacity of disabled child elements (like inputs and labels), resulting in poor contrast and unreadable tooltips. It also makes the container look inconsistent with sibling containers.
**Action:** Remove `opacity-50` from parent wrappers of disabled groups. Instead, apply `disabled:opacity-50` directly to interactive elements (like inputs) and standard `opacity-50` to non-interactive associated elements (like labels) to preserve contrast and accessibility.
