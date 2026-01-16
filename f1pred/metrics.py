from __future__ import annotations
from typing import Dict, Any, Optional
import numpy as np
import pandas as pd
from scipy.stats import spearmanr, kendalltau

def accuracy_top_k(pred_order_ids: list, actual_order_ids: list, k: int = 3) -> float:
    pred_top = set(pred_order_ids[:k])
    act_top = set(actual_order_ids[:k])
    if k == 0:
        return float("nan")
    return len(pred_top & act_top) / float(k)

def brier_pairwise(pairwise_prob: np.ndarray, actual_positions: np.ndarray) -> float:
    """
    Calculate the Brier score for pairwise probability predictions.

    Vectorized implementation: O(n^2) using NumPy broadcasting.
    """
    n = len(actual_positions)
    if n < 2:
        return float("nan")

    # Broadcast actual positions to create a pairwise comparison matrix
    # Y[i, j] = 1.0 if actual[i] < actual[j] (i finished ahead of j) else 0.0
    Y = (actual_positions[:, None] < actual_positions[None, :]).astype(float)

    # Squared error for all pairs
    squared_errors = (pairwise_prob - Y) ** 2

    # Select upper triangle indices (i < j) to count each pair exactly once
    i_indices, j_indices = np.triu_indices(n, k=1)
    relevant_errors = squared_errors[i_indices, j_indices]

    if relevant_errors.size == 0:
        return float("nan")

    return float(np.mean(relevant_errors))

def crps_position(prob_row: np.ndarray, actual_pos: int) -> float:
    N = prob_row.shape[0]
    F = np.cumsum(prob_row)
    H = np.ones(N)
    if actual_pos > 1:
        H[: actual_pos - 1] = 0.0
    return float(np.mean((F - H) ** 2))

def compute_event_metrics(ranked_df: pd.DataFrame,
                          prob_matrix: Optional[np.ndarray],
                          pairwise: Optional[np.ndarray],
                          session: str,
                          season: int,
                          rnd: int) -> Dict[str, Any]:
    df = ranked_df.copy()
    if "driver_id" in df.columns:
        drv_col = "driver_id"
    elif "driverId" in df.columns:
        drv_col = "driverId"
    else:
        drv_col = None

    if "actual_position" not in df.columns or df["actual_position"].isna().all():
        return {
            "season": season, "round": rnd, "event": session,
            "n": int(df.shape[0]),
            "spearman": np.nan, "kendall": np.nan, "accuracy_top3": np.nan,
            "brier_pairwise": np.nan, "crps": np.nan
        }

    dfv = df.dropna(subset=["actual_position"]).copy()
    if dfv.empty:
        return {
            "season": season, "round": rnd, "event": session,
            "n": int(df.shape[0]),
            "spearman": np.nan, "kendall": np.nan, "accuracy_top3": np.nan,
            "brier_pairwise": np.nan, "crps": np.nan
        }

    y_pred = dfv["predicted_position"].to_numpy(dtype=float)
    y_true = dfv["actual_position"].to_numpy(dtype=float)

    try:
        sp_val = float(spearmanr(y_pred, y_true, nan_policy="omit").correlation)
    except Exception:
        sp_val = np.nan
    try:
        kd = float(kendalltau(y_pred, y_true, nan_policy="omit").correlation)
    except Exception:
        kd = np.nan

    pred_order_ids = df.sort_values("predicted_position")[drv_col].tolist() if drv_col else []
    actual_order_ids = df.sort_values("actual_position", na_position="last")[drv_col].tolist() if drv_col else []
    acc_top3 = accuracy_top_k(pred_order_ids, actual_order_ids, k=3) if drv_col else np.nan

    brier_pw = np.nan
    if pairwise is not None and df["actual_position"].notna().all():
        act_pos = df["actual_position"].to_numpy(dtype=int)
        brier_pw = brier_pairwise(pairwise, act_pos)

    crps = np.nan
    if prob_matrix is not None and df["actual_position"].notna().all():
        N = prob_matrix.shape[1]
        if N == df.shape[0]:
            crps_vals = []
            for i in range(N):
                crps_vals.append(crps_position(prob_matrix[i], int(df.iloc[i]["actual_position"])))
            crps = float(np.mean(crps_vals)) if crps_vals else np.nan

    return {
        "season": season,
        "round": rnd,
        "event": session,
        "n": int(df.shape[0]),
        "spearman": sp_val,
        "kendall": kd,
        "accuracy_top3": float(acc_top3) if acc_top3 == acc_top3 else np.nan,
        "brier_pairwise": float(brier_pw) if brier_pw == brier_pw else np.nan,
        "crps": float(crps) if crps == crps else np.nan,
    }