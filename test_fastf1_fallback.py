from datetime import datetime, timezone
from f1pred.data.jolpica import JolpicaClient
from f1pred.data.open_meteo import OpenMeteoClient
from f1pred.config import load_config
from f1pred.features import build_session_features

cfg = load_config("config.yaml")

jc = JolpicaClient(base_url=cfg.data_sources.jolpica.base_url, timeout=cfg.data_sources.jolpica.timeout_seconds)
om = None

print("Fetching features for 2026 Round 1 Race...")
X, meta, roster = build_session_features(
    jc, om, "2026", "1", "race",
    ref_date=datetime.now(timezone.utc),
    cfg=cfg
)

print(f"Features created with shape {X.shape}")
print("Sample of grid and current_quali_pos for Piastri:")
print(X[X["driverId"] == "piastri"][["driverId", "grid", "current_quali_pos"]])
print("Sample of grid and current_quali_pos for Antonelli:")
print(X[X["driverId"] == "antonelli"][["driverId", "grid", "current_quali_pos"]])
