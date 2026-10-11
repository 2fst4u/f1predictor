from __future__ import annotations
import numpy as np


def plackett_luce_scores(scores: np.ndarray, temperature: float = 1.0) -> np.ndarray:
    """
    Convert scores (higher is better) into a probability vector using a
    temperature-controlled softmax. Robust to NaN/Inf and zero variance.

    - If temperature <= 0, it is coerced to a small positive value.
    - Non-finite scores are filled with the mean of finite entries (or 0 if none).
    - If all scores are identical (zero std), returns uniform probabilities.
    """
    scores = np.asarray(scores, dtype=float)
    n = scores.size
    if n == 0:
        return scores  # empty array

    # Replace non-finite values with mean of finite values (or 0 if none)
    finite = np.isfinite(scores)
    if not finite.all():
        if finite.any():
            fill = scores[finite].mean()
        else:
            fill = 0.0
        scores = np.where(finite, scores, fill)

    temp = max(float(temperature), 1e-6)

    # Standardise with numerical safety
    std = scores.std()
    if std < 1e-12:
        s = np.zeros_like(scores)
    else:
        s = (scores - scores.mean()) / (std + 1e-12)
    s = s / temp

    # Stable softmax
    s = s - np.max(s)
    e = np.exp(s)
    denom = e.sum()
    if not np.isfinite(denom) or denom <= 0.0:
        # Fallback: uniform distribution
        return np.full(n, 1.0 / n)

    p = e / denom
    # Enforce exact normalisation (guard tiny fp error)
    return p / p.sum()