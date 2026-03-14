from f1pred.data.fastf1_backend import init_fastf1, get_session_classification
from f1pred.util import get_logger
import logging
import pandas as pd
import numpy as np

# Setup basic logging to see what's happening
logging.basicConfig(level=logging.INFO)
logger = get_logger(__name__)

def check_fastf1_results():
    cache_dir = "fastf1_cache"
    init_fastf1(cache_dir)

    season = 2026
    round_no = 2

    # Names we check in the app
    names_to_try = ["Sprint Qualifying", "SQ", "Sprint Shootout", "Shootout"]

    print(f"Checking 2026 Round 2 (Chinese GP) results in FastF1...")

    import fastf1
    ev = fastf1.get_event(season, round_no)
    for name in names_to_try:
        print(f"\nTrying session name: '{name}'")
        try:
            sess = ev.get_session(name)
            print(f"Loading session '{name}' with laps=True...")
            sess.load(telemetry=False, laps=True, weather=False, messages=False)
            cls = sess.results
            if cls is not None and not cls.empty:
                print(f"SUCCESS: Found results for '{name}'")
                print(f"Shape: {cls.shape}")
                non_null = [c for c in cls.columns if cls[c].notna().any()]
                print(f"Non-null columns: {non_null}")

                # Print first few rows of potentially interesting columns
                cols_to_show = [c for c in ['Abbreviation', 'Position', 'ClassifiedPosition', 'GridPosition', 'Q1', 'Q2', 'Q3', 'Time', 'Status'] if c in cls.columns]
                print(f"Values:\n{cls[cols_to_show].head(10)}")

                # Check if ANY row has a numeric position
                if 'Position' in cls.columns:
                    pos_numeric = pd.to_numeric(cls['Position'], errors='coerce')
                    print(f"Numeric positions found: {pos_numeric.dropna().tolist()}")

                # Check laps
                if sess.laps is not None and not sess.laps.empty:
                    print(f"Laps found: {len(sess.laps)}")
                    best_laps = sess.laps.pick_fastest()
                    if not best_laps.empty:
                         print(f"Best lap found for {len(best_laps)} drivers.")
                         print(f"Top 5 by time:\n{sess.laps.groupby('Abbreviation').LapTime.min().sort_values().head()}")
            else:
                print(f"No results found for '{name}'")
        except Exception as e:
            print(f"Error checking '{name}': {e}")

    return False

if __name__ == "__main__":
    found = check_fastf1_results()
    if not found:
        print("\nCRITICAL: No sprint qualifying results found in FastF1 with any known name.")
    else:
        print("\nResult found.")
