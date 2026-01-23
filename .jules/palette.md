## 2025-05-24 - CLI Spinner with Elapsed Time
**Learning:** Users running long-running CLI tasks (like predictions involving data fetching) benefit significantly from an elapsed time indicator. It reassures them that the process hasn't hung. Using `\033[K` (ANSI clear line) is more robust than calculating backspace length when the line content changes dynamically.
**Action:** Always include an elapsed time counter in CLI spinners for tasks >2s. Use ANSI escape codes for cleaner line clearing.

## 2025-05-24 - CLI Input Validation
**Learning:** For CLI tools, strict input validation with friendly error messages is better than silent fallback to default values, especially for explicit arguments. Silent fallbacks can confuse users who think they are getting results for their specific (invalid) input.
**Action:** Validate CLI arguments early and provide specific, actionable error messages with color coding (Red for errors) to guide the user.

## 2025-05-24 - CLI Visual Hierarchy with Dimming
**Learning:** In CLI applications, using `Style.DIM` for "empty" or inactive parts of a visualization (like the remaining part of a progress bar) creates a cleaner visual hierarchy than using full-brightness characters or spaces. It allows the active data to pop without losing the context of the scale.
**Action:** Use `Style.DIM` for background elements, empty state markers, and secondary information in console output to reduce visual clutter.

## 2025-05-24 - Rich CLI Help Discovery
**Learning:** Users often don't read `README.md` but will try `--help`. Standard `argparse` help is often too dry. Exposing docstring examples via `argparse.RawDescriptionHelpFormatter` and `epilog=__doc__` significantly improves feature discovery right in the terminal.
**Action:** Always configure `argparse` to show usage examples from the module docstring.

## 2025-05-25 - CLI Accessibility for Light Themes
**Learning:** Hardcoding `Fore.WHITE` in CLI applications renders text invisible on terminals with light backgrounds (Light Mode). This is a critical accessibility failure.
**Action:** Use `Fore.RESET` (or `Style.RESET_ALL`) instead of `Fore.WHITE` to respect the user's terminal theme and ensure text contrast.

## 2025-05-25 - CLI Spinner State Clarity
**Learning:** When a step in a CLI process is skipped (e.g., due to missing data), simply hiding it or showing a generic "Success" checkmark is misleading. Users need to distinguish between "Success", "Failure", and "Skipped/Warning" to trust the output.
**Action:** Implement a distinct "Skipped" state (e.g., Yellow `⚠`) in status spinners to provide accurate feedback without interrupting the flow.
