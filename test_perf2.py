import numpy as np
import pandas as pd
import time
from scipy.optimize import minimize

N = 2000
df_calib = pd.DataFrame({
    "season": np.repeat(np.arange(100), 20),
    "round": np.ones(N),
    "gbm_raw": np.random.randn(N),
    "base_form": np.random.randn(N),
    "base_team": np.random.randn(N),
    "base_dt": np.random.randn(N),
    "grid": np.random.randn(N),
    "elo": np.random.randn(N),
    "bt": np.random.randn(N),
    "mixed": np.random.randn(N),
    "actual_pos": np.random.randn(N),
})

def obj_orig(weights):
    wb_gbm, wb_form, wb_tm, wb_dt, wb_grid = weights[0:5]

    wb_gbm = max(0, wb_gbm)
    wb_form = max(0, wb_form)
    wb_tm = max(0, wb_tm)
    wb_dt = max(0, wb_dt)
    wb_grid = max(0, wb_grid)

    raw_pace_series = (wb_gbm * df_calib["gbm_raw"] +
                       wb_form * df_calib["base_form"] +
                       wb_tm * df_calib["base_team"] +
                       wb_dt * df_calib["base_dt"])

    event_keys = df_calib[["season", "round"]].values
    unique_events, event_indices = np.unique(event_keys, axis=0, return_inverse=True)

    pace_final = np.zeros(len(df_calib))

    for i in range(len(unique_events)):
        mask = (event_indices == i)

        ev_pace = raw_pace_series.values[mask]
        mu_p = np.nanmean(ev_pace)
        sd_p = np.nanstd(ev_pace)
        p_z = (ev_pace - mu_p) / (sd_p + 1e-6)

        ev_grid = df_calib["grid"].values[mask]
        mu_g = np.nanmean(ev_grid)
        sd_g = np.nanstd(ev_grid)
        g_z = (ev_grid - mu_g) / (sd_g + 1e-6)

        grid_imp = np.clip(wb_grid, 0.0, 1.0)
        pace_final[mask] = (1.0 - grid_imp) * p_z + grid_imp * g_z

    pace = pace_final

    we_pace, we_elo, we_bt, we_mixed = weights[5:9]
    we_pace = max(0, we_pace)
    we_elo = max(0, we_elo)
    we_bt = max(0, we_bt)
    we_mixed = max(0, we_mixed)

    combined = (we_pace * pace +
                we_elo * df_calib["elo"] +
                we_bt * df_calib["bt"] +
                we_mixed * df_calib["mixed"])

    corr = np.corrcoef(combined, df_calib["actual_pos"])[0, 1]
    return -corr + 0.001 * np.sum(weights**2)

# Precompute for optimized version
arr_gbm_raw = df_calib["gbm_raw"].values
arr_base_form = df_calib["base_form"].values
arr_base_team = df_calib["base_team"].values
arr_base_dt = df_calib["base_dt"].values
arr_elo = df_calib["elo"].values
arr_bt = df_calib["bt"].values
arr_mixed = df_calib["mixed"].values
arr_actual_pos = df_calib["actual_pos"].values

event_keys = df_calib[["season", "round"]].values
unique_events, event_indices = np.unique(event_keys, axis=0, return_inverse=True)

grid_vals = df_calib["grid"].values
g_z_precomputed = np.zeros_like(grid_vals, dtype=float)
event_masks = [np.where(event_indices == i)[0] for i in range(len(unique_events))]

for mask in event_masks:
    ev_grid = grid_vals[mask]
    mu_g = np.nanmean(ev_grid)
    sd_g = np.nanstd(ev_grid)
    g_z_precomputed[mask] = (ev_grid - mu_g) / (sd_g + 1e-6)

def obj_opt(weights):
    wb_gbm, wb_form, wb_tm, wb_dt, wb_grid = weights[0:5]

    wb_gbm = max(0, wb_gbm)
    wb_form = max(0, wb_form)
    wb_tm = max(0, wb_tm)
    wb_dt = max(0, wb_dt)
    wb_grid = max(0, wb_grid)

    raw_pace = (wb_gbm * arr_gbm_raw +
                wb_form * arr_base_form +
                wb_tm * arr_base_team +
                wb_dt * arr_base_dt)

    pace_final = np.zeros(len(raw_pace))
    grid_imp = np.clip(wb_grid, 0.0, 1.0)

    for mask in event_masks:
        ev_pace = raw_pace[mask]
        mu_p = np.nanmean(ev_pace)
        sd_p = np.nanstd(ev_pace)
        p_z = (ev_pace - mu_p) / (sd_p + 1e-6)

        pace_final[mask] = (1.0 - grid_imp) * p_z + grid_imp * g_z_precomputed[mask]

    we_pace, we_elo, we_bt, we_mixed = weights[5:9]
    we_pace = max(0, we_pace)
    we_elo = max(0, we_elo)
    we_bt = max(0, we_bt)
    we_mixed = max(0, we_mixed)

    combined = (we_pace * pace_final +
                we_elo * arr_elo +
                we_bt * arr_bt +
                we_mixed * arr_mixed)

    corr = np.corrcoef(combined, arr_actual_pos)[0, 1]
    return -corr + 0.001 * np.sum(weights**2)

init_w = [0.6, 0.4, 0.3, 0.2, 0.8, 0.4, 0.2, 0.2, 0.2]
bounds = [
    (0.05, 1.0),
    (0.05, 1.0),
    (0.2, 2.0),
    (0.0, 2.0),
    (0.1, 0.95),
    (0.0, 1.0),
    (0.0, 1.0),
    (0.0, 1.0),
    (0.0, 1.0),
]

t0 = time.time()
res_orig = minimize(obj_orig, init_w, bounds=bounds, method='L-BFGS-B')
t_orig = time.time() - t0

t0 = time.time()
res_opt = minimize(obj_opt, init_w, bounds=bounds, method='L-BFGS-B')
t_opt = time.time() - t0

print(f"Original minimization: {t_orig:.3f}s")
print(f"Optimized minimization: {t_opt:.3f}s")
print(f"Speedup: {t_orig/t_opt:.1f}x")
print(f"Equal Results? {np.allclose(res_orig.x, res_opt.x)}")
