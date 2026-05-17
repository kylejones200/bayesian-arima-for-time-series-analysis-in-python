from __future__ import annotations

from dataclasses import dataclass

import numpy as np


@dataclass(frozen=True)
class ForecastMetrics:
    mape: float
    rmse: float
    mae: float
    coverage_95: float | None = None


def mape(actual: np.ndarray, predicted: np.ndarray) -> float:
    actual = np.asarray(actual, dtype=float)
    predicted = np.asarray(predicted, dtype=float)
    mask = actual != 0
    if not np.any(mask):
        return float("nan")
    return float(np.mean(np.abs((actual[mask] - predicted[mask]) / actual[mask])) * 100)


def rmse(actual: np.ndarray, predicted: np.ndarray) -> float:
    err = np.asarray(actual, dtype=float) - np.asarray(predicted, dtype=float)
    return float(np.sqrt(np.mean(err**2)))


def mae(actual: np.ndarray, predicted: np.ndarray) -> float:
    err = np.asarray(actual, dtype=float) - np.asarray(predicted, dtype=float)
    return float(np.mean(np.abs(err)))


def interval_coverage(
    actual: np.ndarray,
    lower: np.ndarray,
    upper: np.ndarray,
) -> float:
    actual = np.asarray(actual, dtype=float)
    lower = np.asarray(lower, dtype=float)
    upper = np.asarray(upper, dtype=float)
    inside = (actual >= lower) & (actual <= upper)
    return float(np.mean(inside))


def summarize_forecast(
    actual: np.ndarray,
    predicted: np.ndarray,
    *,
    lower: np.ndarray | None = None,
    upper: np.ndarray | None = None,
) -> ForecastMetrics:
    coverage = None
    if lower is not None and upper is not None:
        coverage = interval_coverage(actual, lower, upper)
    return ForecastMetrics(
        mape=mape(actual, predicted),
        rmse=rmse(actual, predicted),
        mae=mae(actual, predicted),
        coverage_95=coverage,
    )
