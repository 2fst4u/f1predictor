## 2026-05-04 - Found unlinked form labels
**Learning:** Found an accessibility issue pattern in the app's components where some form labels were not explicitly linked to their corresponding inputs using `for` and `id` attributes. This decreases screen reader support and general accessibility.
**Action:** Applied `for` and `id` attributes to explicitly link labels with their inputs in `index.html`. Look for this pattern in future UI reviews.
