## 2025-05-24 - CLI Spinner with Elapsed Time
**Learning:** Users running long-running CLI tasks (like predictions involving data fetching) benefit significantly from an elapsed time indicator. It reassures them that the process hasn't hung. Using `\033[K` (ANSI clear line) is more robust than calculating backspace length when the line content changes dynamically.
**Action:** Always include an elapsed time counter in CLI spinners for tasks >2s. Use ANSI escape codes for cleaner line clearing.
