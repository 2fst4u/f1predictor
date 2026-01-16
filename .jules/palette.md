## 2025-05-24 - CLI Spinner with Elapsed Time
**Learning:** Users running long-running CLI tasks (like predictions involving data fetching) benefit significantly from an elapsed time indicator. It reassures them that the process hasn't hung. Using `\033[K` (ANSI clear line) is more robust than calculating backspace length when the line content changes dynamically.
**Action:** Always include an elapsed time counter in CLI spinners for tasks >2s. Use ANSI escape codes for cleaner line clearing.

## 2025-05-24 - CLI Input Validation
**Learning:** For CLI tools, strict input validation with friendly error messages is better than silent fallback to default values, especially for explicit arguments. Silent fallbacks can confuse users who think they are getting results for their specific (invalid) input.
**Action:** Validate CLI arguments early and provide specific, actionable error messages with color coding (Red for errors) to guide the user.
