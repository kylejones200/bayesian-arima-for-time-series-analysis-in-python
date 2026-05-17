import numpy as np
import pytest

from bayesian_arima_ts.metrics import interval_coverage, mape, summarize_forecast


def test_mape_ignores_zero_actuals():
    actual = np.array([0.0, 10.0, 20.0])
    predicted = np.array([1.0, 9.0, 22.0])
    assert mape(actual, predicted) == pytest.approx(10.0)


def test_interval_coverage():
    actual = np.array([1.0, 2.0, 3.0])
    lower = np.array([0.5, 1.5, 2.5])
    upper = np.array([1.5, 2.5, 3.5])
    assert interval_coverage(actual, lower, upper) == 1.0


def test_summarize_forecast_includes_coverage():
    actual = np.array([1.0, 2.0])
    predicted = np.array([1.1, 1.9])
    metrics = summarize_forecast(
        actual,
        predicted,
        lower=np.array([0.5, 1.0]),
        upper=np.array([1.5, 3.0]),
    )
    assert metrics.coverage_95 is not None
