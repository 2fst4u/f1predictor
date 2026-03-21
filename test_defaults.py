import sys
from datetime import datetime, timezone
import pandas as pd
from f1pred.config import load_config
from f1pred.data.jolpica import JolpicaClient
from f1pred.data.open_meteo import OpenMeteoClient
from f1pred.features import build_session_features
from f1pred.models import train_pace_model

cfg = load_config("config.yaml")
jc = JolpicaClient("http://ergast.com/api/f1", 30, 0.1)
om = OpenMeteoClient("", "", "", "")

# Try predicting a later race where data is populated
print("Building features...")
X, meta, roster = build_session_features(
    jc, om, 2023, 10, "qualifying", datetime(2023, 7, 8, 14, 0, tzinfo=timezone.utc), cfg
)

print("Checking X columns for NaN...")
for col in X.columns:
    if X[col].isna().all():
        print(f"  {col} is entirely NaN")
    else:
        print(f"  {col} has valid data")
