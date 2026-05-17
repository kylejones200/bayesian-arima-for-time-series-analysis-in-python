#!/usr/bin/env python3
"""Aggregate well-level North Dakota production CSV to monthly state totals."""

from __future__ import annotations

import argparse
from pathlib import Path

import pandas as pd

from bayesian_arima_ts.paths import DEFAULT_DATA_DIR


def aggregate_monthly(
    source: Path,
    *,
    metric: str = "oil",
    out: Path | None = None,
) -> Path:
    col_map = {"oil": "Oil", "gas": "Gas", "gas_sold": "GasSold"}
    if metric not in col_map:
        raise ValueError(f"metric must be one of {list(col_map)}")

    metric_col = col_map[metric]
    frame = pd.read_csv(source, usecols=["ReportDate", metric_col], parse_dates=["ReportDate"])
    monthly = frame.groupby("ReportDate", as_index=False)[metric_col].sum()
    monthly.columns = ["date", metric]

    if out is None:
        out = DEFAULT_DATA_DIR / f"north_dakota_{metric}_monthly.csv"
    out.parent.mkdir(parents=True, exist_ok=True)
    monthly.to_csv(out, index=False)
    print(f"Wrote {len(monthly)} months to {out}")
    return out


def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument(
        "source",
        type=Path,
        nargs="?",
        default=None,
        help="Well-level north_dakota_production.csv (required unless --from-env)",
    )
    parser.add_argument("--metric", choices=["oil", "gas", "gas_sold"], default="oil")
    parser.add_argument("--out", type=Path, default=None)
    args = parser.parse_args()

    source = args.source
    if source is None:
        raise SystemExit(
            "Provide path to well-level CSV, e.g. "
            "scripts/rebuild_monthly.py /path/to/north_dakota_production.csv"
        )

    aggregate_monthly(source, metric=args.metric, out=args.out)


if __name__ == "__main__":
    main()
