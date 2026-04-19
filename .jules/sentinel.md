## 2024-04-12 - Hardcoded JWT Secret Key in Auth Module
**Vulnerability:** The `SECRET_KEY` in `f1pred/auth.py` was hardcoded to a default string `"f1pred-super-secret-key-change-me"`.
**Learning:** Hardcoded default secrets are a critical vulnerability, as they allow anyone to forge valid JWT tokens and bypass authentication.
**Prevention:** Always use environment variables for sensitive secrets and provide a cryptographically secure fallback (e.g., `secrets.token_urlsafe(32)`) if the environment variable is not present, to ensure the application fails securely and avoids deterministic, predictable default secrets.
## 2024-04-19 - Server-Side Request Forgery (SSRF) in Webhook Testing
**Vulnerability:** The `/api/settings/test-webhook` endpoint blindly accepted a user-provided URL in the `WebhookTestRequest` payload and made an HTTP POST request to it using `httpx.AsyncClient` without validating the host or scheme.
**Learning:** This is a classic SSRF vulnerability. An attacker could provide internal network addresses (e.g., `http://169.254.169.254` for cloud metadata, `http://localhost:8080`) or local file paths (`file:///etc/passwd`), tricking the application server into scanning internal ports, reading local files, or making unauthorized internal API calls.
**Prevention:** When an application feature requires sending a webhook to a third-party service based on user input, strictly allowlist the domain or URL prefix (e.g., `https://discord.com/api/webhooks/`). Never allow arbitrary domains, IPs, or non-HTTP(S) schemes.
