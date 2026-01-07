from flask import Flask, render_template, request, redirect, url_for
import os
from f1pred.config import load_config
from f1pred.predict import run_predictions_for_event
from f1pred.util import ensure_dirs, init_caches

app = Flask(__name__)

# Load config
cfg = load_config("config.yaml")

# Initialize caches
ensure_dirs(cfg.paths.output_dir, cfg.paths.reports_dir, cfg.paths.cache_dir, cfg.paths.fastf1_cache)
init_caches(cfg)

@app.route("/", methods=["GET"])
def index():
    return render_template("index.html")

@app.route("/predict", methods=["POST"])
def predict():
    season = request.form.get("season")
    round_val = request.form.get("round")

    # Handle 'current' and 'next' defaults
    if not season:
        season = "current"
    if not round_val:
        round_val = "next"

    sessions = request.form.getlist("sessions")
    if not sessions:
        sessions = ["qualifying", "race"]

    # Run predictions
    # We call run_predictions_for_event with return_results=True to get the data back
    # We set generate_html=False because we will render it dynamically, or maybe we want to keep it?
    # Let's keep it False for now and render on the fly.
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
        return render_template("error.html", message=str(e))

if __name__ == "__main__":
    app.run(debug=True, port=5000)
