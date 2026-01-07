## 2026-01-04 - XSS in HTML Report Generation
**Vulnerability:** The HTML report generation using `jinja2.Template` did not have autoescaping enabled. This means that if any data passed to the template (e.g., driver names, team names, or session titles) contained HTML or script tags, they would be rendered as raw HTML in the browser. This could allow for Cross-Site Scripting (XSS) attacks if the data source is compromised or contains malicious input.
**Learning:** `jinja2.Template` defaults to `autoescape=False`. Explicitly using `jinja2.Environment(autoescape=True)` is necessary to ensure security when generating HTML.
**Prevention:** Always use `jinja2.Environment` with `autoescape=True` when rendering HTML templates, especially when including data from external sources. Avoid using `jinja2.Template` directly for HTML unless you are certain the input is trusted or already escaped.

## 2026-01-04 - Config Validation and DoS Prevention
**Vulnerability:** The application blindly accepted configuration values for refresh intervals and simulation iterations. A malicious or malformed `config.yaml` could cause a Denial of Service (DoS) by setting the refresh rate to be extremely frequent (spamming APIs) or the number of Monte Carlo draws to be excessively high (resource exhaustion).
**Learning:** CLI tools often trust user configuration implicitly, but should validate bounds to prevent accidental misuse or self-inflicted DoS.
**Prevention:** Implement strict validation layers for all configuration inputs, especially numerical bounds (e.g., min/max values) and URL formats, before the application logic begins.

## 2026-01-05 - DoS via Unhandled API Response Types
**Vulnerability:** The application crashed when the external Jolpica API returned non-JSON responses (e.g., 500 HTML error pages). The HTTP client fallback returned the raw string, which downstream code blindly treated as a dictionary, leading to `AttributeError` and application termination.
**Learning:** Never assume external APIs will always return the expected data type, even if the status code is 200 (or if 500s are masked). Defensive coding must handle unexpected types (like `str` instead of `dict`) gracefully.
**Prevention:** Implement strict type checking at the boundary of external data ingestion. Ensure functions like `_extract_mrdata` validate the input type before accessing keys, and return safe default values (like `{}` or `None`) instead of crashing.
