## 2026-01-04 - XSS in HTML Report Generation
**Vulnerability:** The HTML report generation using `jinja2.Template` did not have autoescaping enabled. This means that if any data passed to the template (e.g., driver names, team names, or session titles) contained HTML or script tags, they would be rendered as raw HTML in the browser. This could allow for Cross-Site Scripting (XSS) attacks if the data source is compromised or contains malicious input.
**Learning:** `jinja2.Template` defaults to `autoescape=False`. Explicitly using `jinja2.Environment(autoescape=True)` is necessary to ensure security when generating HTML.
**Prevention:** Always use `jinja2.Environment` with `autoescape=True` when rendering HTML templates, especially when including data from external sources. Avoid using `jinja2.Template` directly for HTML unless you are certain the input is trusted or already escaped.

## 2026-01-05 - DoS via Unhandled API Response Types
**Vulnerability:** The application crashed when the external Jolpica API returned non-JSON responses (e.g., 500 HTML error pages). The HTTP client fallback returned the raw string, which downstream code blindly treated as a dictionary, leading to `AttributeError` and application termination.
**Learning:** Never assume external APIs will always return the expected data type, even if the status code is 200 (or if 500s are masked). Defensive coding must handle unexpected types (like `str` instead of `dict`) gracefully.
**Prevention:** Implement strict type checking at the boundary of external data ingestion. Ensure functions like `_extract_mrdata` validate the input type before accessing keys, and return safe default values (like `{}` or `None`) instead of crashing.

## 2026-01-14 - Terminal Escape Injection
**Vulnerability:** The application printed data from external APIs (driver names, team names) directly to the console. If a compromised API returned ANSI escape codes, it could corrupt terminal output or potentially execute commands in vulnerable terminal emulators (though less common nowadays, it's a valid injection vector).
**Learning:** Even CLI applications need output sanitization. External data is untrusted, regardless of the display medium (HTML vs Terminal).
**Prevention:** Sanitize all external strings using a regex to strip ANSI escape codes before printing to the console, specifically in `StatusSpinner` and prediction output tables.

## 2026-01-18 - DoS via Unbounded Retry-After Header
**Vulnerability:** The HTTP client respected the `Retry-After` header from the Jolpica API without any upper bound. A compromised or misconfigured server could return a massive sleep duration (e.g., 100 years), causing the application to hang indefinitely. This is a Denial of Service (DoS) vector via External Service Interaction.
**Learning:** Never trust control flow instructions from external sources blindly. Even standard headers like `Retry-After` can be weaponized or accidentally malformed to disrupt service availability.
**Prevention:** Implement hard caps on all wait/sleep durations derived from external input. In this case, `Retry-After` was capped at 300 seconds (5 minutes) to ensure the application remains responsive even if the API demands excessive waits.
## 2025-02-18 - Log Injection in Exception Messages
**Vulnerability:** Unsanitized user input (season/round) was directly embedded in `ValueError` messages in data clients. Malicious input containing newlines could forge log entries when these exceptions were logged.
**Learning:** Standard validation raising exceptions can still be a vector for Log Injection if the exception message includes the raw input.
**Prevention:** Always use `repr()` or strict sanitization when including untrusted input in exception messages, even if the input is about to be rejected.
