from __future__ import annotations

import logging
from dataclasses import dataclass
from typing import Any

import arviz as az
import numpy as np
import pymc as pm

logger = logging.getLogger(__name__)


@dataclass(frozen=True)
class BayesianARResult:
    idata: az.InferenceData
    ar_order: int
    train: np.ndarray
    forecast_draws: np.ndarray
    forecast_mean: np.ndarray
    forecast_lower: np.ndarray
    forecast_upper: np.ndarray
    in_sample_mean: np.ndarray
    in_sample_lower: np.ndarray
    in_sample_upper: np.ndarray


def _stack_posterior(idata: az.InferenceData, name: str) -> np.ndarray:
    return idata.posterior[name].stack(sample=("chain", "draw")).values


def _posterior_predictive_band(idata: az.InferenceData, var_name: str = "y") -> tuple[np.ndarray, np.ndarray, np.ndarray]:
    samples = idata.posterior_predictive[var_name].stack(sample=("chain", "draw")).values
    if samples.ndim == 3:
        samples = samples[:, 0, :]
    return (
        samples.mean(axis=0),
        np.percentile(samples, 2.5, axis=0),
        np.percentile(samples, 97.5, axis=0),
    )


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
    state = np.tile(history[-order:], (n_paths, 1))

    for step in range(horizon):
        innovation = rng.normal(0.0, sigma, size=n_paths)
        paths[:, step] = intercept + np.sum(state * ar_coefs, axis=1) + innovation
        state = np.roll(state, -1, axis=1)
        state[:, -1] = paths[:, step]
    return paths


def forecast_from_posterior(
    idata: az.InferenceData,
    history: np.ndarray,
    *,
    horizon: int,
    n_paths: int = 200,
    seed: int = 42,
) -> tuple[np.ndarray, np.ndarray, np.ndarray, np.ndarray]:
    rho_samples = _stack_posterior(idata, "rho")
    sigma_samples = _stack_posterior(idata, "sigma")
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
) -> az.InferenceData:
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
        idata.extend(pm.sample_posterior_predictive(idata, var_names=["y"]))

    summary = az.summary(idata, var_names=["rho", "sigma"])
    logger.info("Bayesian AR posterior summary:\n%s", summary.to_string())
    divergences = int(idata.sample_stats["diverging"].sum())
    if divergences:
        logger.warning("MCMC finished with %s divergences", divergences)
    return idata


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
        chains=int(bayes_cfg.get("chains", 2)),
        target_accept=float(bayes_cfg.get("target_accept", 0.9)),
        cores=int(bayes_cfg.get("cores", 1)),
        random_seed=seed,
    )

    in_sample_mean, in_sample_lower, in_sample_upper = _posterior_predictive_band(idata, "y")
    forecast_draws, forecast_mean, forecast_lower, forecast_upper = forecast_from_posterior(
        idata,
        train,
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
        in_sample_mean=in_sample_mean,
        in_sample_lower=in_sample_lower,
        in_sample_upper=in_sample_upper,
    )
