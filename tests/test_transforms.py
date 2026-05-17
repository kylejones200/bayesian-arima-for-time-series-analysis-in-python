import numpy as np
import pandas as pd
import pytest

from bayesian_arima_ts.transforms import (
    apply_transform,
    prepare_series,
    temporal_train_test_split,
)


def test_temporal_split_preserves_order():
    values = np.arange(10, dtype=float)
    train, test = temporal_train_test_split(values, test_size=3)
    assert len(train) == 7
    assert len(test) == 3
    np.testing.assert_array_equal(test, np.array([7.0, 8.0, 9.0]))


def test_log_diff_shortens_series():
    index = pd.period_range("2020-01", periods=24, freq="M").to_timestamp()
    series = pd.Series(np.linspace(100, 150, 24), index=index)
    prepared = prepare_series(series, "log_diff")
    assert len(prepared.values) == 23


def test_invalid_test_size():
    with pytest.raises(ValueError):
        temporal_train_test_split(np.arange(5.0), test_size=5)
