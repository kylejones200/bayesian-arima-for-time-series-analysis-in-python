import numpy as np

from bayesian_arima_ts.bayesian import _simulate_ar_paths


def test_simulate_ar_paths_shape():
    rng = np.random.default_rng(0)
    rho = np.array([0.0, 0.4, -0.1])
    history = np.array([0.1, -0.05, 0.02], dtype=float)
    paths = _simulate_ar_paths(
        rho,
        sigma=0.5,
        history=history,
        horizon=6,
        n_paths=20,
        rng=rng,
    )
    assert paths.shape == (20, 6)
