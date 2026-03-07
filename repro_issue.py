import pandas as pd
import numpy as np
from f1pred.models import train_pace_model

# Mock X
data = {
    'driverId': [f'd{i}' for i in range(20)],
    'form_index': [20.0, 19.0, 18.0] + [10.0]*17, # d0 is best
    'team_form_index': [10.0]*20,
    'driver_team_form_index': [5.0]*20,
    'grid': [20.0, 1.0, 2.0] + list(range(4, 21)), # d0 (best) starts last (pos 20)
    'constructorId': ['t1']*20
}
X = pd.DataFrame(data)

class Blending:
    gbm_weight = 0.75
    baseline_weight = 0.25
    baseline_team_factor = 0.3
    baseline_driver_team_factor = 0.2

class Modelling:
    blending = Blending()

class DummyCfg:
    modelling = Modelling()

cfg = DummyCfg()

_, pace_hat, _ = train_pace_model(X, "race", cfg)

# Sort by pace (lower is better)
order = np.argsort(pace_hat)
print("Predicted Order:")
for i in order:
    rank = np.where(order==i)[0][0] + 1
    print(f"Pos {rank:2d}: {X.iloc[i]['driverId']} (Form: {X.iloc[i]['form_index']:4.1f}, Grid: {X.iloc[i]['grid']:4.1f}, Pace: {pace_hat[i]:5.2f})")
