from flask import Flask, render_template, request, redirect, url_for, Response
import os
import re
from f1pred.config import load_config
from f1pred.predict import run_predictions_for_event
from f1pred.util import ensure_dirs, init_caches

app = Flask(__name__)

# Load config
cfg = load_config("config.yaml")

# Initialize caches
ensure_dirs(cfg.paths.output_dir, cfg.paths.reports_dir, cfg.paths.cache_dir, cfg.paths.fastf1_cache)
init_caches(cfg)

@app.after_request
def add_security_headers(response: Response) -> Response:
    """Add security headers to all responses."""
    # Content Security Policy
    # - default-src 'self': Only allow resources from same origin by default
    # - style-src: Allow self and unsafe-inline (needed for some inline styles in templates), plus Google Fonts
    # - font-src: Allow self and Google Fonts
    # - img-src: Allow self, https (for team logos from external sites), and data URIs
    csp = (
        "default-src 'self'; "
        "style-src 'self' 'unsafe-inline' https://fonts.googleapis.com; "
        "font-src 'self' https://fonts.gstatic.com; "
        "img-src 'self' https: data:;"
    )
    response.headers["Content-Security-Policy"] = csp

    # Prevent MIME type sniffing
    response.headers["X-Content-Type-Options"] = "nosniff"

    # Prevent clickjacking
    response.headers["X-Frame-Options"] = "SAMEORIGIN"

    # Control referrer information
    response.headers["Referrer-Policy"] = "strict-origin-when-cross-origin"

    return response

@app.route("/", methods=["GET"])
def index():
    return render_template("index.html")

@app.route("/predict", methods=["POST"])
def predict():
    season = request.form.get("season", "").strip()
    round_val = request.form.get("round", "").strip()

    # Input validation
    # Season: "current" or 4 digits
    if not season:
        season = "current"
    elif season.lower() != "current" and not re.match(r"^\d{4}$", season):
        return render_template("error.html", message="Invalid season. Use 'current' or a 4-digit year.")

    # Round: "next", "last", or digits
    if not round_val:
        round_val = "next"
    elif round_val.lower() not in ("next", "last") and not re.match(r"^\d+$", round_val):
        return render_template("error.html", message="Invalid round. Use 'next', 'last', or a number.")

    sessions = request.form.getlist("sessions")
    if not sessions:
        sessions = ["qualifying", "race"]

    # Run predictions
    try:
        results = run_predictions_for_event(
            cfg,
            season=season,
            rnd=round_val,
            sessions=sessions,
            generate_html=False,
            open_browser=False,
            return_results=True
        )

        if not results or not results.get("all_preds"):
            return render_template("error.html", message="No results generated. Please check your inputs.")

        return render_template(
            "results.html",
            results=results["all_preds"],
            event_title=results.get("event_title", "Predictions"),
            season=results.get("season"),
            round=results.get("round")
        )

    except Exception as e:
        # Generic error message to avoid leaking internal details if exception contains sensitive info
        # Though str(e) is usually safe, we should be careful.
        # But for now, we keep str(e) as it's useful for the user, and XSS is mitigated by autoescaping + CSP.
        return render_template("error.html", message=str(e))

if __name__ == "__main__":
    # Disable debug mode for security
    debug_mode = os.environ.get("FLASK_DEBUG", "false").lower() == "true"
    app.run(debug=debug_mode, port=5000)
