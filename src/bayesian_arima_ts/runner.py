from __future__ import annotations

import argparse
import logging
from pathlib import Path
from typing import Any

from bayesian_arima_ts import __version__
from bayesian_arima_ts.bayesian import fit_and_forecast as fit_bayesian
from bayesian_arima_ts.classical import fit_and_forecast as fit_classical
from bayesian_arima_ts.config import configure_logging, load_config
from bayesian_arima_ts.data import load_series
from bayesian_arima_ts.metrics import summarize_forecast
from bayesian_arima_ts.plots import (
    log_metrics_table,
    output_paths,
    plot_cfg,
    plot_forecast_comparison,
    plot_modeling_split,
    plot_raw_series,
    plot_trace,
)
from bayesian_arima_ts.transforms import adf_pvalue, prepare_series, temporal_train_test_split

logger = logging.getLogger(__name__)


def run_pipeline(cfg: dict[str, Any]) -> dict[str, Any]:
    data_cfg = cfg.get("data") or {}
    dataset = str(data_cfg.get("dataset", "airline_passengers"))
    transform = str(data_cfg.get("transform", "log_diff"))
    test_size = int(data_cfg.get("test_size", 24))
    seed = int(data_cfg.get("seed", 42))
    ar_order = int((cfg.get("bayesian") or {}).get("ar_order", 2))

    series = load_series(
        dataset,
        seed=seed,
        csv_path=data_cfg.get("csv_path"),
        value_col=str(data_cfg.get("value_col", "value")),
        date_col=data_cfg.get("date_col"),
    )
    prepared = prepare_series(series, transform)
    pvalue = adf_pvalue(prepared.values)
    logger.info(
        "Loaded %s (%d points). Transform=%s, ADF p-value=%.4f",
        dataset,
        len(prepared.values),
        transform,
        pvalue,
    )
    if pvalue > 0.05:
        logger.warning(
            "Series may still be non-stationary on the modeling scale (ADF p > 0.05). "
            "Consider a stronger differencing transform in config.yaml."
        )

    train, test = temporal_train_test_split(prepared.values, test_size=test_size)
    modeling_index = prepared.modeling.index
    train_index = modeling_index[: len(train)]
    test_index = modeling_index[len(train) :]

    figures_dir, fmt, dpi, show = plot_cfg(cfg)
    paths = output_paths(
        figures_dir,
        fmt,
        ["raw_series", "modeling_split", "mcmc_trace", "holdout_forecasts"],
    )

    plot_raw_series(series, path=paths["raw_series"], dpi=dpi, show=show)
    plot_modeling_split(
        modeling_index,
        train,
        test,
        transform=transform,
        path=paths["modeling_split"],
        dpi=dpi,
        show=show,
    )

    bayesian = fit_bayesian(
        train,
        ar_order=ar_order,
        horizon=len(test),
        cfg=cfg,
    )
    classical = fit_classical(train, horizon=len(test), cfg=cfg)

    if (cfg.get("output") or {}).get("save_trace", True):
        plot_trace(bayesian.idata, path=paths["mcmc_trace"], dpi=dpi, show=show)

    plot_forecast_comparison(
        test_index,
        test,
        bayesian.forecast_mean,
        bayesian.forecast_lower,
        bayesian.forecast_upper,
        classical.forecast,
        classical.forecast_lower,
        classical.forecast_upper,
        path=paths["holdout_forecasts"],
        dpi=dpi,
        show=show,
    )

    bayes_metrics = summarize_forecast(
        test,
        bayesian.forecast_mean,
        lower=bayesian.forecast_lower,
        upper=bayesian.forecast_upper,
    )
    classical_metrics = summarize_forecast(
        test,
        classical.forecast,
        lower=classical.forecast_lower,
        upper=classical.forecast_upper,
    )
    log_metrics_table(bayes_metrics, classical_metrics)

    return {
        "dataset": dataset,
        "transform": transform,
        "adf_pvalue": pvalue,
        "bayesian_metrics": bayes_metrics,
        "classical_metrics": classical_metrics,
        "arima_order": classical.order,
        "figures": {k: str(v) for k, v in paths.items()},
    }


def build_parser() -> argparse.ArgumentParser:
    parser = argparse.ArgumentParser(
        description="Bayesian AR (PyMC) vs classical auto-ARIMA on real time series data.",
    )
    parser.add_argument(
        "--config",
        type=Path,
        default=None,
        help="Path to config.yaml (default: project config.yaml)",
    )
    parser.add_argument(
        "--dataset",
        choices=["airline_passengers", "co2", "sunspots", "synthetic", "csv"],
        help="Override data.dataset from config",
    )
    parser.add_argument("--version", action="version", version=f"%(prog)s {__version__}")
    return parser


def main(argv: list[str] | None = None) -> None:
    parser = build_parser()
    args = parser.parse_args(argv)

    cfg = load_config(args.config)
    configure_logging(cfg)

    if args.dataset:
        cfg.setdefault("data", {})["dataset"] = args.dataset

    logger.info("Starting pipeline (bayesian-arima-ts %s)", __version__)
    run_pipeline(cfg)


if __name__ == "__main__":
    main()
