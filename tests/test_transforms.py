import numpy as np
import pandas as pd
import pytest

from bayesian_arima_ts.transforms import (
    apply_transform,
    modeling_to_levels,
    prepare_series,
    temporal_train_test_split,
)


def test_log_diff_reduces_length():
    index = pd.period_range("2000-01", periods=24, freq="M").to_timestamp()
    series = pd.Series(np.linspace(100, 130, 24), index=index)
    prepared = prepare_series(series, "log_diff")
    assert len(prepared.values) == 23


def test_temporal_split_sizes():
    values = np.arange(50, dtype=float)
    train, test = temporal_train_test_split(values, test_size=10)
    assert len(train) == 40
    assert len(test) == 10


def test_modeling_to_levels_log_diff():
    anchor = 110.0
    diffs = np.array([np.log(121 / 110), np.log(133.1 / 121)])
    recovered = modeling_to_levels(diffs, anchor, "log_diff")
    assert recovered[0] == pytest.approx(121.0, rel=1e-5)
    assert recovered[1] == pytest.approx(133.1, rel=1e-5)


def test_unknown_transform_raises():
    series = pd.Series([1.0, 2.0, 3.0])
    with pytest.raises(ValueError, match="Unknown transform"):
        apply_transform(series, "invalid")
