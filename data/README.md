# Data

## North Dakota production (default)

The committed default is `data/north_dakota_oil_monthly.csv`: state-wide **barrels of oil per month** (107 months, Jan 2016–Nov 2024), aggregated from the PPDM well-level file.

To use the full well-level archive locally (without committing your path):

```bash
cp config.local.yaml.example config.local.yaml
# Edit production_path in config.local.yaml

# Or one-off:
uv run bayesian-arima-run --production-path /path/to/north_dakota_production.csv
```

`load_north_dakota_production()` detects well-level files (`API_WELLNO`, `Oil`, `ReportDate`, …) and sums by month. Set `production_metric: gas` or `gas_sold` for other columns.

## Custom CSV

Set `data.dataset: csv` in `config.yaml` and place a file at `data/series.csv`:

```csv
date,value
2020-01-01,120.5
2020-02-01,118.2
```

Override columns with `data.value_col` and `data.date_col`. Use monthly (or regular) timestamps when seasonality matters.

## Transform cheat sheet

| Pattern | Suggested `data.transform` |
|---------|---------------------------|
| Strong trend | `log_diff` or `diff` |
| Monthly seasonality | `log_diff` + seasonal ARIMA (classical side) |
| Already stationary residuals | `none` |
