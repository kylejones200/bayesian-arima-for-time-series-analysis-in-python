from __future__ import annotations

from pathlib import Path

import numpy as np
import pandas as pd
from statsmodels.datasets import co2, sunspots
from statsmodels.datasets.airpassengers import load_pandas as load_airpassengers

from bayesian_arima_ts.paths import DEFAULT_DATA_DIR

TransformName = str
DatasetName = str


def load_airline_passengers() -> pd.Series:
    frame = load_airpassengers().data
    series = frame["passengers"].astype(float)
    series.index = pd.period_range("1949-01", periods=len(series), freq="M").to_timestamp()
    series.name = "passengers"
    return series


def load_co2_series() -> pd.Series:
    frame = co2.load_pandas().data
    series = frame["co2"].astype(float).dropna()
    series.index = pd.to_datetime(series.index)
    series.name = "co2"
    return series


def load_sunspots() -> pd.Series:
    frame = sunspots.load_pandas().data
    series = frame["sunspots"].astype(float)
    series.index = pd.to_datetime(frame["YEAR"].astype(int), format="%Y")
    series.name = "sunspots"
    return series


def load_synthetic_trend(*, seed: int = 42, n: int = 144) -> pd.Series:
    rng = np.random.default_rng(seed)
    index = pd.period_range("2010-01", periods=n, freq="M").to_timestamp()
    seasonal = 8 * np.sin(2 * np.pi * np.arange(n) / 12)
    trend = 0.35 * np.arange(n)
    noise = rng.normal(0, 2.5, n)
    values = 100 + trend + seasonal + noise
    return pd.Series(values, index=index, name="synthetic")


def load_series_from_csv(path: Path | str, *, value_col: str, date_col: str | None = None) -> pd.Series:
    frame = pd.read_csv(path)
    if date_col:
        index = pd.to_datetime(frame[date_col])
    else:
        index = pd.RangeIndex(len(frame))
    series = frame[value_col].astype(float)
    series.index = index
    series.name = Path(path).stem
    return series.sort_index()


def load_series(
    dataset: DatasetName,
    *,
    csv_path: Path | str | None = None,
    value_col: str = "value",
    date_col: str | None = "date",
    seed: int = 42,
) -> pd.Series:
    if dataset == "airline_passengers":
        return load_airline_passengers()
    if dataset == "co2":
        return load_co2_series()
    if dataset == "sunspots":
        return load_sunspots()
    if dataset == "synthetic":
        return load_synthetic_trend(seed=seed)
    if dataset == "csv":
        if csv_path is None:
            csv_path = DEFAULT_DATA_DIR / "series.csv"
        return load_series_from_csv(csv_path, value_col=value_col, date_col=date_col)
    raise ValueError(
        f"Unknown dataset {dataset!r}. "
        "Choose airline_passengers, co2, sunspots, synthetic, or csv."
    )
