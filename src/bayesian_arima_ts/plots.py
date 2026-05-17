from __future__ import annotations

from pathlib import Path
from typing import Any

import arviz as az
import matplotlib.pyplot as plt
import numpy as np
import pandas as pd

from bayesian_arima_ts.metrics import ForecastMetrics


def _save_or_show(path: Path | None, *, dpi: int, show: bool) -> None:
    if path is not None:
        path.parent.mkdir(parents=True, exist_ok=True)
        plt.savefig(path, dpi=dpi, bbox_inches="tight")
    if show:
        plt.show()
    plt.close()


def plot_raw_series(
    series: pd.Series,
    *,
    path: Path | None,
    dpi: int,
    show: bool,
) -> None:
    fig, ax = plt.subplots(figsize=(10, 5))
    series.plot(ax=ax, color="steelblue", linewidth=1.5)
    ax.set_title(f"Observed series: {series.name or 'value'}")
    ax.set_xlabel("Date")
    ax.set_ylabel("Level")
    ax.grid(True, alpha=0.3)
    fig.tight_layout()
    _save_or_show(path, dpi=dpi, show=show)


def plot_modeling_split(
    modeling_index: pd.Index,
    train: np.ndarray,
    test: np.ndarray,
    *,
    transform: str,
    path: Path | None,
    dpi: int,
    show: bool,
) -> None:
    split = len(train)
    fig, ax = plt.subplots(figsize=(10, 5))
    ax.plot(modeling_index[:split], train, label="Train", color="steelblue")
    ax.plot(modeling_index[split:], test, label="Holdout", color="darkorange")
    ax.set_title(f"Modeling scale ({transform})")
    ax.set_xlabel("Date")
    ax.set_ylabel("Transformed value")
    ax.legend()
    ax.grid(True, alpha=0.3)
    fig.tight_layout()
    _save_or_show(path, dpi=dpi, show=show)


def plot_trace(
    idata: az.InferenceData,
    *,
    path: Path | None,
    dpi: int,
    show: bool,
) -> None:
    az.plot_trace(idata, var_names=["rho", "sigma"])
    fig = plt.gcf()
    fig.tight_layout()
    _save_or_show(path, dpi=dpi, show=show)


def plot_forecast_comparison(
    test_index: pd.Index,
    test: np.ndarray,
    bayes_mean: np.ndarray,
    bayes_lower: np.ndarray,
    bayes_upper: np.ndarray,
    classical_mean: np.ndarray,
    classical_lower: np.ndarray,
    classical_upper: np.ndarray,
    *,
    path: Path | None,
    dpi: int,
    show: bool,
) -> None:
    fig, ax = plt.subplots(figsize=(11, 6))
    ax.plot(test_index, test, label="Actual", color="black", linewidth=2, marker="o", markersize=4)

    ax.plot(test_index, bayes_mean, label="Bayesian AR mean", color="crimson")
    ax.fill_between(test_index, bayes_lower, bayes_upper, color="crimson", alpha=0.2, label="Bayesian 95%")

    ax.plot(test_index, classical_mean, label="Auto ARIMA", color="royalblue", linestyle="--")
    ax.fill_between(
        test_index,
        classical_lower,
        classical_upper,
        color="royalblue",
        alpha=0.15,
        label="ARIMA 95%",
    )

    ax.set_title("Holdout forecast comparison (modeling scale)")
    ax.set_xlabel("Date")
    ax.set_ylabel("Value")
    ax.legend(loc="best")
    ax.grid(True, alpha=0.3)
    fig.tight_layout()
    _save_or_show(path, dpi=dpi, show=show)


def log_metrics_table(
    bayesian: ForecastMetrics,
    classical: ForecastMetrics,
) -> None:
    import logging

    logger = logging.getLogger(__name__)
    logger.info(
        "Holdout metrics (modeling scale)\n"
        "  Model          MAPE     RMSE      MAE   95%% cov\n"
        "  Bayesian AR   %6.2f%%  %7.3f  %7.3f  %6.1f%%\n"
        "  Auto ARIMA      %6.2f%%  %7.3f  %7.3f  %6.1f%%",
        bayesian.mape,
        bayesian.rmse,
        bayesian.mae,
        (bayesian.coverage_95 or 0) * 100,
        classical.mape,
        classical.rmse,
        classical.mae,
        (classical.coverage_95 or 0) * 100,
    )


def output_paths(figures_dir: Path, fmt: str, names: list[str]) -> dict[str, Path]:
    return {name: figures_dir / f"{name}.{fmt}" for name in names}


def plot_level_forecasts(
    test_index: pd.Index,
    actual_level: np.ndarray,
    bayes_level: np.ndarray,
    arima_level: np.ndarray,
    *,
    y_label: str = "Level",
    path: Path | None,
    dpi: int,
    show: bool,
) -> None:
    fig, ax = plt.subplots(figsize=(11, 6))
    ax.plot(test_index, actual_level, label="Actual", color="black", linewidth=2, marker="o", markersize=4)
    ax.plot(test_index, bayes_level, label="Bayesian AR", color="crimson")
    ax.plot(test_index, arima_level, label="Auto ARIMA", color="royalblue", linestyle="--")
    ax.set_title("Holdout forecasts (original units)")
    ax.set_xlabel("Date")
    ax.set_ylabel(y_label)
    ax.legend(loc="best")
    ax.grid(True, alpha=0.3)
    fig.tight_layout()
    _save_or_show(path, dpi=dpi, show=show)


def plot_posterior_predictive(
    train_index: pd.Index,
    train: np.ndarray,
    idata: az.InferenceData,
    *,
    path: Path | None,
    dpi: int,
    show: bool,
) -> None:
    if "posterior_predictive" not in idata:
        return
    y_ppc = idata["posterior_predictive"]["y"].stack(sample=("chain", "draw")).values
    if y_ppc.ndim > 2:
        y_ppc = y_ppc.reshape(y_ppc.shape[0], -1)
    mean = y_ppc.mean(axis=0)
    lower = np.percentile(y_ppc, 2.5, axis=0)
    upper = np.percentile(y_ppc, 97.5, axis=0)

    fig, ax = plt.subplots(figsize=(10, 5))
    ax.plot(train_index, train, label="Observed", color="black", linewidth=1.2)
    ax.plot(train_index, mean, label="PPC mean", color="crimson", alpha=0.8)
    ax.fill_between(train_index, lower, upper, color="crimson", alpha=0.2, label="95% PPC")
    ax.set_title("Posterior predictive check (training window)")
    ax.set_xlabel("Date")
    ax.set_ylabel("Modeling scale")
    ax.legend()
    ax.grid(True, alpha=0.3)
    fig.tight_layout()
    _save_or_show(path, dpi=dpi, show=show)


def plot_cfg(cfg: dict[str, Any]) -> tuple[Path, str, int, bool]:
    output = cfg.get("output") or {}
    figures_dir = Path(output.get("figures_dir", "outputs/figures"))
    fmt = str(output.get("figure_format", "png"))
    dpi = int(output.get("figure_dpi", 120))
    show = bool(output.get("show", False))
    return figures_dir, fmt, dpi, show
