1. **Add Alpine component for keyboard tab navigation**
   - Use Alpine.js to handle `ArrowRight` and `ArrowLeft` on `tablist` elements to allow switching between tabs, following WAI-ARIA guidelines for keyboard accessibility.
2. **Apply keyboard navigation to Main Tabs**
   - Add `@keydown.right` and `@keydown.left` to the main navigation tabs (`Predictions` / `Settings`) in `f1pred/templates/index.html` to switch `activeTab`.
3. **Apply keyboard navigation to Session Tabs**
   - Add keyboard navigation support for the session tabs (`Sprint Shootout`, `Sprint`, `Qualifying`, `Race`) which switch `activeSession`.
4. **Pre-commit step**
   - Run verification scripts.
