from f1pred.data.fastf1_backend import init_fastf1, get_session_classification
from f1pred.predict import _get_actual_positions_for_session
from f1pred.data.jolpica import JolpicaClient
import pandas as pd
import numpy as np

def debug_mapping():
    init_fastf1("fastf1_cache")
    jc = JolpicaClient("https://api.jolpi.ca/ergast/f1")

    # 2026 R2 Chinese GP
    season = 2026
    round_no = 2

    # Roster as derive_roster would return it (roughly)
    # I'll just use the IDs I know for 2026
    roster_view = pd.DataFrame([
        {"driverId": "russell", "number": "63", "code": "RUS", "name": "George Russell"},
        {"driverId": "antonelli", "number": "12", "code": "ANT", "name": "Kimi Antonelli"},
        {"driverId": "norris", "number": "1", "code": "NOR", "name": "Lando Norris"},
        {"driverId": "piastri", "number": "81", "code": "PIA", "name": "Oscar Piastri"},
        {"driverId": "leclerc", "number": "16", "code": "LEC", "name": "Charles Leclerc"},
    ])

    print("Testing _get_actual_positions_for_session for sprint_qualifying...")
    try:
        results = _get_actual_positions_for_session(jc, season, round_no, "sprint_qualifying", roster_view)
    except Exception as e:
        import traceback
        traceback.print_exc()
        results = None

    print(f"\nResulting Series:\n{results}")
    if results is not None:
        print(f"Non-NaN count: {results.count()}")
        print(f"Isna all: {results.isna().all()}")

if __name__ == "__main__":
    import logging
    logging.basicConfig(level=logging.DEBUG)
    debug_mapping()
