from __future__ import annotations

import logging
from dataclasses import dataclass
from typing import Any

import numpy as np
import pmdarima as pmd

logger = logging.getLogger(__name__)


@dataclass(frozen=True)
class ClassicalARIMAResult:
    model: pmd.arima.ARIMA
    order: tuple[int, int, int]
    seasonal_order: tuple[int, int, int, int] | None
    forecast: np.ndarray
    forecast_lower: np.ndarray
    forecast_upper: np.ndarray


def fit_auto_arima(train: np.ndarray, cfg: dict[str, Any]) -> pmd.arima.ARIMA:
    classical = cfg.get("classical") or {}
    seasonal = bool(classical.get("seasonal", True))
    kwargs: dict[str, Any] = {
        "error_action": classical.get("error_action", "ignore"),
        "suppress_warnings": bool(classical.get("suppress_warnings", True)),
        "trace": bool(classical.get("trace", False)),
        "maxiter": int(classical.get("maxiter", 50)),
        "seasonal": seasonal,
        "max_p": int(classical.get("max_p", 3)),
        "max_q": int(classical.get("max_q", 3)),
        "max_d": int(classical.get("max_d", 2)),
    }
    if seasonal:
        kwargs.update(
            m=int(classical.get("m", 12)),
            max_P=int(classical.get("max_P", 2)),
            max_Q=int(classical.get("max_Q", 2)),
            max_D=int(classical.get("max_D", 1)),
        )

    model = pmd.auto_arima(train, **kwargs)
    logger.info(
        "Auto ARIMA selected order=%s seasonal_order=%s AIC=%.2f",
        model.order,
        model.seasonal_order,
        model.aic(),
    )
    return model


def forecast_auto_arima(
    model: pmd.arima.ARIMA,
    horizon: int,
    *,
    alpha: float = 0.05,
) -> tuple[np.ndarray, np.ndarray, np.ndarray]:
    forecast, conf_int = model.predict(
        n_periods=horizon,
        return_conf_int=True,
        alpha=alpha,
    )
    return (
        np.asarray(forecast, dtype=float),
        np.asarray(conf_int[:, 0], dtype=float),
        np.asarray(conf_int[:, 1], dtype=float),
    )


def fit_fixed_ar(
    train: np.ndarray,
    *,
    ar_order: int,
    cfg: dict[str, Any],
) -> pmd.arima.ARIMA:
    classical = cfg.get("classical") or {}
    model = pmd.ARIMA(
        order=(ar_order, 0, 0),
        seasonal_order=(0, 0, 0, 0),
        suppress_warnings=bool(classical.get("suppress_warnings", True)),
        with_intercept=True,
    )
    model.fit(train)
    logger.info(
        "Fixed AR(%d) MLE — AIC=%.2f params=%s",
        ar_order,
        model.aic(),
        model.params(),
    )
    return model


def fit_and_forecast(
    train: np.ndarray,
    *,
    horizon: int,
    cfg: dict[str, Any],
    ar_order: int | None = None,
) -> ClassicalARIMAResult:
    classical = cfg.get("classical") or {}
    mode = str(classical.get("mode", "auto"))

    if mode == "fixed_ar":
        if ar_order is None:
            ar_order = int((cfg.get("bayesian") or {}).get("ar_order", 1))
        model = fit_fixed_ar(train, ar_order=ar_order, cfg=cfg)
    else:
        model = fit_auto_arima(train, cfg)

    forecast, lower, upper = forecast_auto_arima(model, horizon)
    seasonal_order = model.seasonal_order if any(model.seasonal_order) else None
    return ClassicalARIMAResult(
        model=model,
        order=tuple(model.order),
        seasonal_order=seasonal_order,
        forecast=forecast,
        forecast_lower=lower,
        forecast_upper=upper,
    )


def fit_auto_and_fixed_ar(
    train: np.ndarray,
    *,
    horizon: int,
    ar_order: int,
    cfg: dict[str, Any],
) -> tuple[ClassicalARIMAResult, ClassicalARIMAResult]:
    """Auto SARIMA plus fixed AR(p) MLE for apples-to-apples comparison with Bayesian AR(p)."""
    auto_cfg = {**cfg, "classical": {**(cfg.get("classical") or {}), "mode": "auto"}}
    fixed_cfg = {**cfg, "classical": {**(cfg.get("classical") or {}), "mode": "fixed_ar"}}
    auto = fit_and_forecast(train, horizon=horizon, cfg=auto_cfg)
    fixed = fit_and_forecast(train, horizon=horizon, cfg=fixed_cfg, ar_order=ar_order)
    return auto, fixed
