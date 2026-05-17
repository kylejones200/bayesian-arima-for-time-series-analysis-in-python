from bayesian_arima_ts.data import load_airline_passengers, load_synthetic_trend


def test_airline_passengers_length():
    series = load_airline_passengers()
    assert len(series) == 144
    assert series.name == "passengers"


def test_synthetic_is_monthly():
    series = load_synthetic_trend(seed=0)
    assert len(series) == 144
    assert series.index.freqstr in (None, "M", "ME")
