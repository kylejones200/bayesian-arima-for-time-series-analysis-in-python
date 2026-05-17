import numpy as np
import pytest

from bayesian_arima_ts.metrics import interval_coverage, summarize_forecast


def test_interval_coverage():
    actual = np.array([1.0, 2.0, 3.0])
    lower = np.array([0.5, 1.5, 2.5])
    upper = np.array([1.5, 2.5, 2.9])
    assert interval_coverage(actual, lower, upper) == pytest.approx(2 / 3)


def test_summarize_forecast():
    actual = np.array([10.0, 20.0, 30.0])
    pred = np.array([11.0, 18.0, 33.0])
    metrics = summarize_forecast(actual, pred)
    assert metrics.rmse > 0
    assert metrics.mape > 0
