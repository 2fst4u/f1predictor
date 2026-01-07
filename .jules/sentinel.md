## 2026-01-07 - Security Hardening of Flask App
**Vulnerability:** The Flask application was running with `debug=True` by default, lacked input validation for `season`/`round` parameters, and was missing standard security headers (CSP, X-Frame-Options, etc.).
**Learning:** `Flask.run(debug=True)` should never be default in code that might be deployed or exposed. Security headers must be explicitly added as Flask does not provide them by default. Input validation at the controller layer prevents exceptions and potential DoS vectors deeper in the stack.
**Prevention:**
1. Use `os.environ.get("FLASK_DEBUG")` to toggle debug mode.
2. Implement an `after_request` hook to inject security headers.
3. Validate all inputs against allowlists or regex patterns before passing them to business logic.
