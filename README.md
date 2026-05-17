# Bayesian ARIMA for time series analysis in Python

Published: 2024-12-29  
Medium: [Bayesian ARIMA for time series analysis in Python](https://medium.com/@kyle-t-jones/bayesian-arima-for-time-series-analysis-in-python-aabbfe41dcf0)

Robust comparison of a **Bayesian AR model** (PyMC) and **auto-ARIMA** (pmdarima) on real benchmark series with temporal train/test evaluation, uncertainty bands, and reproducible `uv` tooling.

## Quick start

```bash
cd bayesian-arima-for-time-series-analysis-in-python
uv sync
uv run bayesian-arima-run
```

Figures are written to `outputs/figures/`. Edit `config.yaml` to change dataset, transform, or sampler settings.

## Recommended data

| Dataset (`config.yaml`) | Why use it |
|-------------------------|------------|
| `airline_passengers` (default) | Classic monthly seasonality; shows when differencing + seasonal ARIMA beats a low-order Bayesian AR |
| `co2` | Smooth trend + annual cycle; good for `log_diff` or `seasonal_diff` |
| `sunspots` | Non-seasonal cyclic dynamics; simpler stationary behavior |
| `synthetic` | Controlled AR-like process for debugging |
| `csv` | Your own columnar series via `data/csv_path`, `value_col`, `date_col` |

For business-style forecasting, point `csv` at monthly demand, sales, or utilization with a datetime column.

## Project layout

- `src/bayesian_arima_ts/` — data loading, transforms, PyMC AR, auto-ARIMA, metrics, plots
- `config.yaml` — dataset, transforms, sampler, output paths
- `article.md` — original Medium export

## Tests

```bash
uv sync --extra dev
uv run pytest -m "not slow"
uv run pytest -m slow   # optional MCMC smoke test
```
