from __future__ import annotations

import logging
from dataclasses import dataclass
from pathlib import Path
from typing import Any

import arviz as az
import numpy as np
import pymc as pm

from bayesian_arima_ts.paths import PROJECT_ROOT

logger = logging.getLogger(__name__)


@dataclass(frozen=True)
class BayesianARResult:
    idata: Any
    ar_order: int
    train: np.ndarray
    forecast_draws: np.ndarray
    forecast_mean: np.ndarray
    forecast_lower: np.ndarray
    forecast_upper: np.ndarray
    trace_path: Path | None = None


def _stack_posterior(idata: Any, name: str, *, param_dim: int | None = None) -> np.ndarray:
    """Stack chains/draws to (n_samples, n_params) with sample on axis 0."""
    stacked = idata["posterior"][name].stack(sample=("chain", "draw"))
    other_dims = [dim for dim in stacked.dims if dim != "sample"]
    if other_dims:
        stacked = stacked.transpose("sample", *other_dims)
    values = np.asarray(stacked.values, dtype=float)
    n_samples = values.shape[0]
    if values.ndim == 1:
        return values if param_dim != 1 else values.reshape(-1, 1)
    if param_dim is not None:
        return values.reshape(n_samples, param_dim)
    return values.reshape(n_samples, -1)


def _simulate_ar_paths(
    rho: np.ndarray,
    sigma: float,
    history: np.ndarray,
    *,
    horizon: int,
    n_paths: int,
    rng: np.random.Generator,
) -> np.ndarray:
    intercept = float(rho[0])
    ar_coefs = rho[1:]
    order = len(ar_coefs)
    paths = np.zeros((n_paths, horizon))
    # Columns are [y_{t-1}, y_{t-2}, ...] to match PyMC pm.AR(rho=[c, phi1, phi2, ...])
    state = np.tile(history[-order:][::-1], (n_paths, 1))
    for step in range(horizon):
        innovation = rng.normal(0.0, sigma, size=n_paths)
        paths[:, step] = intercept + np.sum(state * ar_coefs, axis=1) + innovation
        state = np.roll(state, 1, axis=1)
        state[:, 0] = paths[:, step]
    return paths


def forecast_from_posterior(
    idata: Any,
    history: np.ndarray,
    *,
    ar_order: int,
    horizon: int,
    n_paths: int = 200,
    seed: int = 42,
) -> tuple[np.ndarray, np.ndarray, np.ndarray, np.ndarray]:
    rho_size = ar_order + 1
    rho_samples = _stack_posterior(idata, "rho", param_dim=rho_size)
    sigma_samples = _stack_posterior(idata, "sigma", param_dim=1).ravel()
    n_draws = rho_samples.shape[0]
    rng = np.random.default_rng(seed)
    all_paths = np.zeros((n_draws, n_paths, horizon))
    for draw_idx in range(n_draws):
        all_paths[draw_idx] = _simulate_ar_paths(
            rho_samples[draw_idx],
            float(sigma_samples[draw_idx]),
            history,
            horizon=horizon,
            n_paths=n_paths,
            rng=rng,
        )

    flat = all_paths.reshape(-1, horizon)
    mean = flat.mean(axis=0)
    lower = np.percentile(flat, 2.5, axis=0)
    upper = np.percentile(flat, 97.5, axis=0)
    return flat, mean, lower, upper


def fit_bayesian_ar(
    train: np.ndarray,
    *,
    ar_order: int,
    draws: int,
    tune: int,
    chains: int,
    target_accept: float,
    cores: int,
    random_seed: int,
) -> Any:
    rho_size = ar_order + 1
    init_dist = pm.Normal.dist(0.0, 5.0, shape=ar_order)
    with pm.Model():
        rho = pm.Normal("rho", mu=0.0, sigma=0.5, shape=rho_size)
        sigma = pm.HalfNormal("sigma", sigma=1.0)
        pm.AR(
            "y",
            rho=rho,
            sigma=sigma,
            constant=True,
            init_dist=init_dist,
            observed=train,
        )
        idata = pm.sample(
            draws=draws,
            tune=tune,
            chains=chains,
            target_accept=target_accept,
            cores=cores,
            random_seed=random_seed,
            return_inferencedata=True,
        )
        ppc = pm.sample_posterior_predictive(idata, var_names=["y"], extend_inferencedata=False)
        idata["posterior_predictive"] = ppc["posterior_predictive"]

    summary = az.summary(idata, var_names=["rho", "sigma"])
    logger.info("Bayesian AR posterior summary:\n%s", summary.to_string())
    diverging = idata["sample_stats"]["diverging"] if "sample_stats" in idata else None
    divergences = int(diverging.sum()) if diverging is not None else 0
    if divergences:
        logger.warning("MCMC finished with %s divergences", divergences)
    return idata


def _resolve_trace_path(bayes_cfg: dict[str, Any]) -> Path | None:
    raw = bayes_cfg.get("trace_path")
    if not raw:
        return None
    path = Path(raw)
    return path if path.is_absolute() else PROJECT_ROOT / path


def fit_and_forecast(
    train: np.ndarray,
    *,
    ar_order: int,
    horizon: int,
    cfg: dict[str, Any],
) -> BayesianARResult:
    bayes_cfg = cfg.get("bayesian") or {}
    seed = int(bayes_cfg.get("random_seed", 42))
    idata = fit_bayesian_ar(
        train,
        ar_order=ar_order,
        draws=int(bayes_cfg.get("draws", 1000)),
        tune=int(bayes_cfg.get("tune", 1000)),
        chains=int(bayes_cfg.get("chains", 4)),
        target_accept=float(bayes_cfg.get("target_accept", 0.9)),
        cores=int(bayes_cfg.get("cores", 1)),
        random_seed=seed,
    )
    trace_path = _resolve_trace_path(bayes_cfg)
    if trace_path is not None:
        trace_path.parent.mkdir(parents=True, exist_ok=True)
        try:
            idata.to_netcdf(trace_path)
            logger.info("Saved MCMC trace to %s", trace_path)
        except (ValueError, OSError) as exc:
            logger.warning("Could not save trace to %s: %s", trace_path, exc)

    forecast_draws, forecast_mean, forecast_lower, forecast_upper = forecast_from_posterior(
        idata,
        train,
        ar_order=ar_order,
        horizon=horizon,
        seed=seed + 1,
    )
    return BayesianARResult(
        idata=idata,
        ar_order=ar_order,
        train=train,
        forecast_draws=forecast_draws,
        forecast_mean=forecast_mean,
        forecast_lower=forecast_lower,
        forecast_upper=forecast_upper,
        trace_path=trace_path,
    )
