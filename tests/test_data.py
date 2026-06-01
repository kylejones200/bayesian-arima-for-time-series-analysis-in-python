import pytest

from bayesian_arima_ts.data import load_series


def test_north_dakota_monthly_aggregate():
    series = load_series("north_dakota_production")
    assert len(series) >= 100
    assert series.name.startswith("nd_")
    assert series.index.is_monotonic_increasing


@pytest.mark.parametrize("dataset", ["airline_passengers", "co2", "sunspots", "synthetic"])
def test_builtin_datasets_load(dataset: str):
    series = load_series(dataset, seed=42)
    assert len(series) > 24
    assert series.notna().all()


def test_unknown_dataset_raises():
    with pytest.raises(ValueError, match="Unknown dataset"):
        load_series("not_a_dataset")
