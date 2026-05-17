from __future__ import annotations

import argparse
import logging
from pathlib import Path
from typing import Any

import numpy as np

from bayesian_arima_ts import __version__
from bayesian_arima_ts.bayesian import fit_and_forecast as fit_bayesian
from bayesian_arima_ts.classical import fit_and_forecast as fit_classical
from bayesian_arima_ts.classical import fit_auto_and_fixed_ar
from bayesian_arima_ts.comparison import (
    build_comparison_report,
    log_comparison_report,
    save_comparison_report,
)
from bayesian_arima_ts.config import configure_logging, load_config
from bayesian_arima_ts.data import load_series
from bayesian_arima_ts.metrics import summarize_forecast
from bayesian_arima_ts.paths import PROJECT_ROOT
from bayesian_arima_ts.plots import (
    log_metrics_table,
    output_paths,
    plot_cfg,
    plot_forecast_comparison,
    plot_level_forecasts,
    plot_modeling_split,
    plot_posterior_predictive,
    plot_raw_series,
    plot_trace,
)
from bayesian_arima_ts.transforms import (
    PreparedSeries,
    adf_pvalue,
    level_anchor,
    modeling_to_levels,
    prepare_series,
    temporal_train_test_split,
)

logger = logging.getLogger(__name__)


def apply_quick_mode(cfg: dict[str, Any]) -> dict[str, Any]:
    quick = cfg.get("quick") or {}
    bayesian = dict(cfg.get("bayesian") or {})
    bayesian.update(
        {
            "draws": int(quick.get("draws", 200)),
            "tune": int(quick.get("tune", 200)),
            "chains": int(quick.get("chains", 2)),
        }
    )
    return {**cfg, "bayesian": bayesian}


def _level_metrics(
    prepared: PreparedSeries,
    train_len: int,
    transform: str,
    test: np.ndarray,
    bayesian_mean: np.ndarray,
    classical_mean: np.ndarray,
) -> dict[str, Any]:
    anchor = level_anchor(prepared, train_len)
    actual = modeling_to_levels(test, anchor, transform)
    bayes = modeling_to_levels(bayesian_mean, anchor, transform)
    arima = modeling_to_levels(classical_mean, anchor, transform)
    return {
        "actual": actual,
        "bayesian": bayes,
        "arima": arima,
        "bayesian_metrics": summarize_forecast(actual, bayes),
        "arima_metrics": summarize_forecast(actual, arima),
    }


def _y_label_for_dataset(dataset: str, production_metric: str) -> str:
    if dataset == "north_dakota_production":
        unit = "MCF" if production_metric == "gas" else "barrels"
        return f"Production ({unit})"
    return "Level"


def run_pipeline(cfg: dict[str, Any]) -> dict[str, Any]:
    data_cfg = cfg.get("data") or {}
    dataset = str(data_cfg.get("dataset", "airline_passengers"))
    transform = str(data_cfg.get("transform", "log_diff"))
    test_size = int(data_cfg.get("test_size", 24))
    seed = int(data_cfg.get("seed", 42))
    production_metric = str(data_cfg.get("production_metric", "oil"))
    ar_order = int((cfg.get("bayesian") or {}).get("ar_order", 2))

    series = load_series(
        dataset,
        seed=seed,
        csv_path=data_cfg.get("csv_path"),
        value_col=str(data_cfg.get("value_col", "value")),
        date_col=data_cfg.get("date_col"),
        production_path=data_cfg.get("production_path"),
        production_metric=production_metric,
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
    test_index = modeling_index[len(train) :]

    figures_dir, fmt, dpi, show = plot_cfg(cfg)
    paths = output_paths(
        figures_dir,
        fmt,
        [
            "raw_series",
            "modeling_split",
            "mcmc_trace",
            "posterior_predictive",
            "holdout_forecasts",
            "holdout_forecasts_level",
        ],
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
    compare_fixed = bool((cfg.get("classical") or {}).get("compare_fixed_ar", True))
    fixed_ar = None
    fixed_ar_metrics = None
    if compare_fixed:
        classical, fixed_ar = fit_auto_and_fixed_ar(
            train, horizon=len(test), ar_order=ar_order, cfg=cfg
        )
    else:
        classical = fit_classical(train, horizon=len(test), cfg=cfg)

    if (cfg.get("output") or {}).get("save_trace", True):
        plot_trace(bayesian.idata, path=paths["mcmc_trace"], dpi=dpi, show=show)

    plot_posterior_predictive(
        modeling_index[: len(train)],
        train,
        bayesian.idata,
        path=paths["posterior_predictive"],
        dpi=dpi,
        show=show,
    )

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

    level = _level_metrics(
        prepared, len(train), transform, test, bayesian.forecast_mean, classical.forecast
    )
    y_label = _y_label_for_dataset(dataset, production_metric)
    plot_level_forecasts(
        test_index,
        level["actual"],
        level["bayesian"],
        level["arima"],
        y_label=y_label,
        path=paths["holdout_forecasts_level"],
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
    if fixed_ar is not None:
        fixed_ar_metrics = summarize_forecast(
            test,
            fixed_ar.forecast,
            lower=fixed_ar.forecast_lower,
            upper=fixed_ar.forecast_upper,
        )
    log_metrics_table(bayes_metrics, classical_metrics)
    logger.info(
        "Holdout metrics (original units) — Bayesian RMSE: %.2f | ARIMA RMSE: %.2f",
        level["bayesian_metrics"].rmse,
        level["arima_metrics"].rmse,
    )

    report = build_comparison_report(
        dataset=dataset,
        transform=transform,
        adf_pvalue=pvalue,
        train=train,
        test=test,
        bayesian=bayesian,
        classical=classical,
        bayesian_metrics=bayes_metrics,
        arima_metrics=classical_metrics,
        fixed_ar=fixed_ar,
        fixed_ar_metrics=fixed_ar_metrics,
        level_metrics={
            "bayesian": level["bayesian_metrics"],
            "arima": level["arima_metrics"],
        },
    )
    log_comparison_report(report)

    results_path = Path((cfg.get("output") or {}).get("results_path", "outputs/results.json"))
    if not results_path.is_absolute():
        results_path = PROJECT_ROOT / results_path
    save_comparison_report(report, results_path)

    return {
        "dataset": dataset,
        "transform": transform,
        "adf_pvalue": pvalue,
        "bayesian_metrics": bayes_metrics,
        "classical_metrics": classical_metrics,
        "arima_order": classical.order,
        "figures": {k: str(v) for k, v in paths.items()},
        "results_path": str(results_path),
        "mcmc_r_hat_max": report.mcmc.r_hat_max,
        "forecast_correlation": report.forecast_comparison.correlation,
    }


def build_parser() -> argparse.ArgumentParser:
    parser = argparse.ArgumentParser(
        description=(
            "Bayesian AR(p) with PyMC vs classical auto-ARIMA. "
            "Models are fit on a transformed (often differenced) scale; "
            "see holdout_forecasts_level for original units."
        ),
    )
    parser.add_argument(
        "--config",
        type=Path,
        default=None,
        help="Path to config.yaml (default: project config.yaml)",
    )
    parser.add_argument(
        "--dataset",
        choices=[
            "airline_passengers",
            "co2",
            "sunspots",
            "synthetic",
            "north_dakota_production",
            "csv",
        ],
        help="Override data.dataset from config",
    )
    parser.add_argument(
        "--production-path",
        type=Path,
        default=None,
        help="Override data.production_path (north_dakota_production)",
    )
    parser.add_argument(
        "--production-metric",
        choices=["oil", "gas", "gas_sold"],
        default=None,
        help="Override data.production_metric (oil, gas, gas_sold)",
    )
    parser.add_argument(
        "--quick",
        action="store_true",
        help="Fast MCMC (uses quick.* settings in config.yaml)",
    )
    parser.add_argument("--version", action="version", version=f"%(prog)s {__version__}")
    return parser


def main(argv: list[str] | None = None) -> None:
    parser = build_parser()
    args = parser.parse_args(argv)

    cfg = load_config(args.config)
    configure_logging(cfg)

    if args.quick:
        cfg = apply_quick_mode(cfg)
        logger.info("Quick mode: reduced MCMC draws/tune/chains")

    if args.dataset:
        cfg.setdefault("data", {})["dataset"] = args.dataset
    if args.production_path:
        cfg.setdefault("data", {})["production_path"] = str(args.production_path)
    if args.production_metric:
        cfg.setdefault("data", {})["production_metric"] = args.production_metric

    logger.info("Starting pipeline (bayesian-arima-ts %s)", __version__)
    run_pipeline(cfg)


if __name__ == "__main__":
    main()
