## 2024-04-12 - Hardcoded JWT Secret Key in Auth Module
**Vulnerability:** The `SECRET_KEY` in `f1pred/auth.py` was hardcoded to a default string `"f1pred-super-secret-key-change-me"`.
**Learning:** Hardcoded default secrets are a critical vulnerability, as they allow anyone to forge valid JWT tokens and bypass authentication.
**Prevention:** Always use environment variables for sensitive secrets and provide a cryptographically secure fallback (e.g., `secrets.token_urlsafe(32)`) if the environment variable is not present, to ensure the application fails securely and avoids deterministic, predictable default secrets.
