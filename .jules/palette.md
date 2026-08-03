## 2025-02-23 - Auth Form Accessibility & Screen Reader Improvements
**Learning:** Decorative font-icons used as loading spinners (like `fa-spin`) are often announced as confusing gibberish by screen readers if they lack `aria-hidden="true"`. Furthermore, password manager UX is severely degraded if `autocomplete` attributes (like `username`, `current-password`, and `new-password`) are omitted from authentication and password change forms.
**Action:** When adding or auditing forms and loading states in Alpine.js/Tailwind components, explicitly verify that all `<i class="fas ...">` decorative icons include `aria-hidden="true"`, and all standard authentication inputs have WCAG 1.3.5 compliant `autocomplete` attributes.

## 2025-02-24 - Disabled Form Label Contrast
**Learning:** Applying `opacity-50` to form labels (e.g., for disabled fields) against a dark background (like `bg-gray-800`) compounds colors and causes the contrast ratio to fail WCAG AA standards (dropping below 4.5:1).
**Action:** Never use `opacity-50` on `<label>` elements for disabled inputs. Keep standard text colors (e.g. `text-gray-400`) and rely on `disabled:opacity-50` or `disabled:cursor-not-allowed` on the `<input>` element itself to convey the disabled state visually.

## 2024-05-18 - [Dynamic State Announcement & Form Types]
**Learning:** Screen readers need explicit text changes to announce loading states dynamically when `aria-label` is static, which can be accomplished by using a visually hidden `<span class="sr-only">` that toggles text via Alpine.js (`x-text`). Additionally, using `type="url"` over `type="text"` for webhook/API inputs natively triggers the correct mobile keyboard with '.com' and '/' shortcuts.
**Action:** Default to `sr-only` dynamic spans instead of static `aria-label` for buttons with changing active states (e.g., loading). Always enforce correct HTML5 input types for strings representing specific data structures (URLs, emails, tel).
## 2025-02-18 - Accessibility for Icon-only Buttons
**Learning:** For dynamic icon-only buttons (like password visibility toggles with Alpine `x-text`), standard `:aria-label` bindings can be problematic as screen readers may not consistently announce their state changes, and automated translation tools often ignore them. Using an inner visually hidden `<span class="sr-only">` element whose text changes dynamically is a more robust accessibility pattern.
**Action:** When implementing icon-only buttons with dynamic states, prefer using an inner `.sr-only` text span rather than relying solely on `aria-label` attributes to ensure robust screen reader announcements and better internationalization support.
