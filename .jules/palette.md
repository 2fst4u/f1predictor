## 2026-01-08 - Accessible Tooltips for Data Tables
**Learning:** Adding title attributes to table headers provides a simple, accessible way to explain complex metrics (like 'Pred pos (μ)') without cluttering the UI. Pairing this with a dotted underline (CSS) provides a subtle visual cue that more info is available, improving discoverability.
**Action:** Use 'th[title]' with 'text-decoration: underline dotted' as a standard pattern for abbreviated column headers in reports.

## 2026-01-08 - CLI Visualizations
**Learning:** Adding simple ASCII/Unicode bar charts (e.g., `███··`) next to percentage values in console output significantly improves scanability and allows users to quickly identify dominant probabilities without parsing numbers.
**Action:** Use inline block character bars for probability columns in CLI tables where space permits.

## 2026-01-08 - Visual Scanning in CLI
**Learning:** Adding emojis (🌧️, ☀️, 🌡️) alongside text labels (e.g., "Rain:", "Wind:") reduces cognitive load and allows users to parse weather conditions faster in dense CLI outputs, without losing the clarity of explicit labels.
**Action:** Use standard weather emojis to augment text labels in CLI summaries.
