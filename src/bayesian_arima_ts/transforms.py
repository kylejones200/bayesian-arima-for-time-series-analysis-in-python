from __future__ import annotations

from dataclasses import dataclass

import numpy as np
import pandas as pd
from statsmodels.tsa.stattools import adfuller


@dataclass(frozen=True)
class PreparedSeries:
    """Modeling scale (often differenced) with metadata to map back to levels."""

    values: np.ndarray
    transform: str
    original: pd.Series
    modeling: pd.Series


def apply_transform(series: pd.Series, transform: str) -> pd.Series:
    if transform == "none":
        return series.astype(float)
    if transform == "log":
        return np.log(series.astype(float))
    if transform == "diff":
        return series.astype(float).diff().dropna()
    if transform == "log_diff":
        return np.log(series.astype(float)).diff().dropna()
    if transform == "seasonal_diff":
        return series.astype(float).diff(12).dropna()
    if transform == "log_seasonal_diff":
        return np.log(series.astype(float)).diff(12).dropna()
    raise ValueError(
        f"Unknown transform {transform!r}. "
        "Use none, log, diff, log_diff, seasonal_diff, or log_seasonal_diff."
    )


def prepare_series(series: pd.Series, transform: str) -> PreparedSeries:
    modeling = apply_transform(series, transform)
    values = modeling.to_numpy(dtype=float)
    return PreparedSeries(
        values=values,
        transform=transform,
        original=series.astype(float),
        modeling=modeling,
    )


def temporal_train_test_split(
    values: np.ndarray, *, test_size: int
) -> tuple[np.ndarray, np.ndarray]:
    if test_size <= 0:
        raise ValueError("test_size must be positive")
    if test_size >= len(values):
        raise ValueError("test_size must be smaller than series length")
    split = len(values) - test_size
    return values[:split], values[split:]


def adf_pvalue(values: np.ndarray) -> float:
    result = adfuller(values, autolag="AIC")
    return float(result[1])
