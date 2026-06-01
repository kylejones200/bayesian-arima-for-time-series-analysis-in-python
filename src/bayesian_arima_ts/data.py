from __future__ import annotations

from pathlib import Path

import numpy as np
import pandas as pd
from statsmodels.datasets import co2, get_rdataset, sunspots

from bayesian_arima_ts.paths import DEFAULT_DATA_DIR

TransformName = str
DatasetName = str


def load_airline_passengers() -> pd.Series:
    frame = get_rdataset("AirPassengers", "datasets").data
    series = frame["value"].astype(float)
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
    value_col = "SUNACTIVITY" if "SUNACTIVITY" in frame.columns else "sunspots"
    series = frame[value_col].astype(float)
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


def load_north_dakota_production(
    path: Path | str | None = None,
    *,
    metric: str = "oil",
) -> pd.Series:
    """
    Load monthly North Dakota production (barrels oil or MCF gas).
    Accepts either the bundled monthly aggregate (``data/north_dakota_oil_monthly.csv``)
    or the full well-level CSV from the PPDM archive (aggregated by ``ReportDate``).
    """
    metric = metric.lower()
    col_map = {"oil": "Oil", "gas": "Gas", "gas_sold": "GasSold"}
    if metric not in col_map:
        raise ValueError(f"Unknown production_metric {metric!r}. Use oil, gas, or gas_sold.")

    csv_path = Path(path) if path else DEFAULT_DATA_DIR / "north_dakota_oil_monthly.csv"
    if not csv_path.is_file():
        raise FileNotFoundError(
            f"North Dakota production file not found: {csv_path}. "
            "Set data.production_path in config.yaml or add data/north_dakota_oil_monthly.csv."
        )

    header = pd.read_csv(csv_path, nrows=0).columns.tolist()
    if "API_WELLNO" in header or "WellName" in header:
        metric_col = col_map[metric]
        date_col = "ReportDate" if "ReportDate" in header else "Date"
        frame = pd.read_csv(csv_path, usecols=[date_col, metric_col], parse_dates=[date_col])
        series = frame.groupby(date_col)[metric_col].sum().sort_index()
        series.name = f"nd_{metric}"
        return series.astype(float)

    value_col = metric if metric in header else {"oil": "oil", "gas": "gas", "gas_sold": "gas_sold"}[metric]
    date_col = "date" if "date" in header else "ReportDate"
    series = load_series_from_csv(csv_path, value_col=value_col, date_col=date_col)
    series.name = f"nd_{metric}"
    return series


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
    production_path: Path | str | None = None,
    production_metric: str = "oil",
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
    if dataset == "north_dakota_production":
        return load_north_dakota_production(production_path, metric=production_metric)
    if dataset == "csv":
        if csv_path is None:
            csv_path = DEFAULT_DATA_DIR / "series.csv"
        return load_series_from_csv(csv_path, value_col=value_col, date_col=date_col)
    raise ValueError(
        f"Unknown dataset {dataset!r}. "
        "Choose airline_passengers, co2, sunspots, synthetic, north_dakota_production, or csv."
    )
