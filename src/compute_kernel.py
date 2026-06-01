"""AR(1) path simulation (numpy reference)."""

from __future__ import annotations

import numpy as np


def simulate_ar1_paths(
    y0: float,
    intercept: float,
    phi: float,
    sigma: float,
    horizon: int,
    n_paths: int,
    seed: int = 42,
) -> np.ndarray:
    rng = np.random.default_rng(seed)
    out = np.empty(n_paths * horizon, dtype=float)
    idx = 0
    for _ in range(n_paths):
        y = float(y0)
        for _ in range(horizon):
            y = intercept + phi * y + sigma * rng.standard_normal()
            out[idx] = y
            idx += 1
    return out
