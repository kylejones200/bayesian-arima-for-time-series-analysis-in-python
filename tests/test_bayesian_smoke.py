import numpy as np
import pytest

from bayesian_arima_ts.bayesian import fit_and_forecast


@pytest.mark.slow
def test_bayesian_ar_smoke():
    rng = np.random.default_rng(0)
    n = 80
    noise = rng.normal(0, 0.4, n)
    y = np.zeros(n)
    for t in range(2, n):
        y[t] = 0.6 * y[t - 1] - 0.2 * y[t - 2] + noise[t]

    cfg = {
        "bayesian": {
            "draws": 60,
            "tune": 60,
            "chains": 1,
            "cores": 1,
            "random_seed": 0,
        }
    }
    result = fit_and_forecast(y[:60], ar_order=2, horizon=10, cfg=cfg)
    assert result.forecast_mean.shape == (10,)
    assert result.forecast_lower.shape == (10,)
