from f1pred.config import load_config
from f1pred.predict import run_predictions_for_event
from f1pred.util import configure_logging
import json

def test():
    configure_logging("ERROR")
    cfg = load_config("config.yaml")
    # Use a past event for speed and reliability
    results = run_predictions_for_event(cfg, season="2023", rnd="1", sessions=["race"], return_results=True)
    # Print a small part of it
    if results:
        # We just want to see the structure
        print(json.dumps({
            "season": results["season"],
            "round": results["round"],
            "sessions": list(results["sessions"].keys())
        }))
        race_results = results["sessions"]["race"]
        print("Columns in ranked:", race_results["ranked"].columns.tolist())
    else:
        print("No results")

if __name__ == "__main__":
    test()
