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

# Manually load the full history to make sure features aren't empty
from f1pred.features import collect_historical_results
now = datetime.now(timezone.utc)
hist = collect_historical_results(jc, 2023, datetime(2023, 7, 8, 14, 0, tzinfo=timezone.utc))

print("Training model...")
try:
    from f1pred.models import build_hist_training_X
    hist_X = build_hist_training_X(hist, X, datetime(2023, 7, 8, 14, 0, tzinfo=timezone.utc))
    pipe, pace_hat, used_features, shap_vals = train_pace_model(X, "qualifying", cfg, hist_X=hist_X)
    print("SUCCESS!")
    print("Used Features:", used_features)
    print("SHAP returned:", shap_vals is not None)
    if shap_vals:
        print("First driver SHAP:", shap_vals[0])
except Exception as e:
    import traceback
    traceback.print_exc()
