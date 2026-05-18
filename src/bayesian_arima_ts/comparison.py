from __future__ import annotations

import json
import logging
from dataclasses import asdict, dataclass, replace
from datetime import UTC, datetime
from pathlib import Path
from typing import Any

import arviz as az
import numpy as np

from bayesian_arima_ts.bayesian import BayesianARResult
from bayesian_arima_ts.classical import ClassicalARIMAResult
from bayesian_arima_ts.metrics import ForecastMetrics

logger = logging.getLogger(__name__)


@dataclass(frozen=True)
class MCMCDiagnostics:
    divergences: int
    r_hat_max: float
    ess_bulk_min: float
    ess_tail_min: float


@dataclass(frozen=True)
class PosteriorSummary:
    rho_mean: list[float]
    rho_sd: list[float]
    sigma_mean: float
    sigma_sd: float


@dataclass(frozen=True)
class ForecastComparison:
    correlation: float
    mean_abs_diff: float
    max_abs_diff: float
    rmse_diff: float
    bayesian_better_rmse: bool


@dataclass(frozen=True)
class ComparisonReport:
    generated_at: str
    dataset: str
    transform: str
    adf_pvalue: float
    train_size: int
    test_size: int
    bayesian_ar_order: int
    arima_order: tuple[int, int, int]
    arima_seasonal_order: tuple[int, int, int, int] | None
    arima_aic: float
    mcmc: MCMCDiagnostics
    posterior: PosteriorSummary
    holdout_actual: list[float]
    bayesian_forecast: list[float]
    arima_forecast: list[float]
    bayesian_metrics: ForecastMetrics
    arima_metrics: ForecastMetrics
    forecast_comparison: ForecastComparison
    fixed_ar_order: tuple[int, int, int] | None = None
    fixed_ar_aic: float | None = None
    fixed_ar_forecast: list[float] | None = None
    fixed_ar_metrics: ForecastMetrics | None = None
    bayes_vs_fixed_ar: ForecastComparison | None = None
    interpretation: dict[str, str] | None = None
    mle_rho: list[float] | None = None
    level_metrics: dict[str, ForecastMetrics] | None = None


def extract_posterior_summary(idata: Any, ar_order: int) -> PosteriorSummary:
    from bayesian_arima_ts.bayesian import _stack_posterior

    rho_size = ar_order + 1
    rho = _stack_posterior(idata, "rho", param_dim=rho_size)
    sigma = _stack_posterior(idata, "sigma", param_dim=1).ravel()
    return PosteriorSummary(
        rho_mean=rho.mean(axis=0).tolist(),
        rho_sd=rho.std(axis=0).tolist(),
        sigma_mean=float(sigma.mean()),
        sigma_sd=float(sigma.std()),
    )


def mcmc_diagnostics(idata: Any) -> MCMCDiagnostics:
    summary = az.summary(idata, var_names=["rho", "sigma"])
    diverging = idata["sample_stats"]["diverging"] if "sample_stats" in idata else None
    divergences = int(diverging.sum()) if diverging is not None else 0
    ess_bulk_col = "ess_bulk" if "ess_bulk" in summary.columns else "ess_mean"
    ess_tail_col = "ess_tail" if "ess_tail" in summary.columns else ess_bulk_col
    return MCMCDiagnostics(
        divergences=divergences,
        r_hat_max=float(summary["r_hat"].max()),
        ess_bulk_min=float(summary[ess_bulk_col].min()),
        ess_tail_min=float(summary[ess_tail_col].min()),
    )


def compare_forecasts(
    bayesian_mean: np.ndarray,
    arima_mean: np.ndarray,
    actual: np.ndarray,
) -> ForecastComparison:
    bayesian_mean = np.asarray(bayesian_mean, dtype=float)
    arima_mean = np.asarray(arima_mean, dtype=float)
    actual = np.asarray(actual, dtype=float)
    diff = bayesian_mean - arima_mean
    bayes_rmse = float(np.sqrt(np.mean((actual - bayesian_mean) ** 2)))
    arima_rmse = float(np.sqrt(np.mean((actual - arima_mean) ** 2)))
    corr = float(np.corrcoef(bayesian_mean, arima_mean)[0, 1]) if len(bayesian_mean) > 1 else float("nan")
    return ForecastComparison(
        correlation=corr,
        mean_abs_diff=float(np.mean(np.abs(diff))),
        max_abs_diff=float(np.max(np.abs(diff))),
        rmse_diff=float(np.sqrt(np.mean(diff**2))),
        bayesian_better_rmse=bayes_rmse < arima_rmse,
    )


def build_interpretation(report: ComparisonReport) -> dict[str, str]:
    """Plain-language summary for results.json."""
    lines = [
        "Bayesian AR(p) on a pre-transformed series is compared to auto-SARIMA and "
        "to a fixed-order AR(p) MLE fit (pmdarima) with the same p.",
        f"MCMC: {report.mcmc.divergences} divergences, max r_hat={report.mcmc.r_hat_max:.3f}, "
        f"min ESS_bulk={report.mcmc.ess_bulk_min:.0f}.",
    ]
    if report.mcmc.r_hat_max > 1.01:
        lines.append("Warning: r_hat > 1.01 — consider more chains or tuning.")
    if report.bayes_vs_fixed_ar is not None:
        lines.append(
            f"Bayesian vs fixed AR({report.bayesian_ar_order}) MLE holdout forecast "
            f"correlation={report.bayes_vs_fixed_ar.correlation:.3f} "
            f"(values near 1 mean MCMC and MLE agree on the same model class)."
        )
    if report.forecast_comparison.bayesian_better_rmse:
        lines.append(
            f"On the modeling scale, Bayesian AR beat auto-ARIMA "
            f"{report.arima_order} (RMSE {report.bayesian_metrics.rmse:.4f} vs "
            f"{report.arima_metrics.rmse:.4f})."
        )
    else:
        lines.append(
            f"On the modeling scale, auto-ARIMA "
            f"{report.arima_order} seasonal={report.arima_seasonal_order} "
            f"beat Bayesian AR (RMSE {report.arima_metrics.rmse:.4f} vs "
            f"{report.bayesian_metrics.rmse:.4f}). Seasonal structure often favors SARIMA."
        )
    return {
        "summary": " ".join(lines[:2]),
        "mcmc": lines[1] if len(lines) > 1 else "",
        "bayes_vs_mle": next((x for x in lines if "fixed AR" in x), ""),
        "holdout_winner": next((x for x in lines if "beat" in x.lower()), ""),
    }


def extract_mle_rho(fixed_ar: ClassicalARIMAResult | None) -> list[float] | None:
    if fixed_ar is None:
        return None
    params = np.asarray(fixed_ar.model.params(), dtype=float)
    if len(params) < 3:
        return None
    return [float(params[0]), float(params[1]), float(params[2])]


def build_comparison_report(
    *,
    dataset: str,
    transform: str,
    adf_pvalue: float,
    train: np.ndarray,
    test: np.ndarray,
    bayesian: BayesianARResult,
    classical: ClassicalARIMAResult,
    bayesian_metrics: ForecastMetrics,
    arima_metrics: ForecastMetrics,
    fixed_ar: ClassicalARIMAResult | None = None,
    fixed_ar_metrics: ForecastMetrics | None = None,
    level_metrics: dict[str, ForecastMetrics] | None = None,
) -> ComparisonReport:
    mcmc = mcmc_diagnostics(bayesian.idata)
    posterior = extract_posterior_summary(bayesian.idata, bayesian.ar_order)
    forecast_cmp = compare_forecasts(
        bayesian.forecast_mean,
        classical.forecast,
        test,
    )
    seasonal = classical.seasonal_order
    bayes_vs_fixed = None
    fixed_order = None
    fixed_aic = None
    fixed_forecast = None
    if fixed_ar is not None and fixed_ar_metrics is not None:
        bayes_vs_fixed = compare_forecasts(
            bayesian.forecast_mean,
            fixed_ar.forecast,
            test,
        )
        fixed_order = fixed_ar.order
        fixed_aic = float(fixed_ar.model.aic())
        fixed_forecast = fixed_ar.forecast.tolist()

    report = ComparisonReport(
        generated_at=datetime.now(UTC).isoformat(),
        dataset=dataset,
        transform=transform,
        adf_pvalue=adf_pvalue,
        train_size=len(train),
        test_size=len(test),
        bayesian_ar_order=bayesian.ar_order,
        arima_order=classical.order,
        arima_seasonal_order=seasonal,
        arima_aic=float(classical.model.aic()),
        mcmc=mcmc,
        posterior=posterior,
        holdout_actual=test.tolist(),
        bayesian_forecast=bayesian.forecast_mean.tolist(),
        arima_forecast=classical.forecast.tolist(),
        bayesian_metrics=bayesian_metrics,
        arima_metrics=arima_metrics,
        forecast_comparison=forecast_cmp,
        fixed_ar_order=fixed_order,
        fixed_ar_aic=fixed_aic,
        fixed_ar_forecast=fixed_forecast,
        fixed_ar_metrics=fixed_ar_metrics,
        bayes_vs_fixed_ar=bayes_vs_fixed,
        interpretation=None,
        mle_rho=extract_mle_rho(fixed_ar),
        level_metrics=level_metrics,
    )
    return replace(report, interpretation=build_interpretation(report))


def _json_default(obj: Any) -> Any:
    if isinstance(obj, np.ndarray):
        return obj.tolist()
    if isinstance(obj, np.generic):
        return obj.item()
    if isinstance(obj, tuple):
        return list(obj)
    raise TypeError(f"Object of type {type(obj)} is not JSON serializable")


def save_comparison_report(report: ComparisonReport, path: Path) -> Path:
    path.parent.mkdir(parents=True, exist_ok=True)
    payload = asdict(report)
    with path.open("w", encoding="utf-8") as handle:
        json.dump(payload, handle, indent=2, default=_json_default)
    logger.info("Wrote comparison report to %s", path)
    return path


def log_comparison_report(report: ComparisonReport) -> None:
    logger.info(
        "MCMC diagnostics: divergences=%d r_hat_max=%.4f ess_bulk_min=%.0f",
        report.mcmc.divergences,
        report.mcmc.r_hat_max,
        report.mcmc.ess_bulk_min,
    )
    if report.mcmc.r_hat_max > 1.01:
        logger.warning("Some parameters have r_hat > 1.01 — consider more tuning/draws")
    if report.mcmc.divergences > 0:
        logger.warning("MCMC reported %d divergent transitions", report.mcmc.divergences)

    logger.info(
        "Posterior rho (intercept + AR): mean=%s",
        [round(v, 4) for v in report.posterior.rho_mean],
    )
    logger.info(
        "Holdout forecast agreement: corr=%.3f mean|diff|=%.4f max|diff|=%.4f",
        report.forecast_comparison.correlation,
        report.forecast_comparison.mean_abs_diff,
        report.forecast_comparison.max_abs_diff,
    )
    logger.info(
        "Holdout RMSE — Bayesian AR: %.4f | Auto ARIMA: %.4f | Bayesian wins: %s",
        report.bayesian_metrics.rmse,
        report.arima_metrics.rmse,
        report.forecast_comparison.bayesian_better_rmse,
    )
    if report.bayes_vs_fixed_ar is not None and report.fixed_ar_metrics is not None:
        logger.info(
            "Same-order AR(%d) MLE — RMSE: %.4f | forecast corr vs Bayesian: %.3f",
            report.bayesian_ar_order,
            report.fixed_ar_metrics.rmse,
            report.bayes_vs_fixed_ar.correlation,
        )
    if report.mle_rho is not None:
        logger.info(
            "Coefficients — posterior rho mean=%s | MLE rho=%s",
            [round(v, 4) for v in report.posterior.rho_mean],
            [round(v, 4) for v in report.mle_rho],
        )
    if report.interpretation:
        for key, text in report.interpretation.items():
            if text:
                logger.info("Interpretation [%s]: %s", key, text)
