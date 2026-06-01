import numpy as np
import pytest

from bayesian_arima_ts.bayesian import fit_bayesian_ar


@pytest.mark.slow
def test_fit_bayesian_ar_smoke():
    train = np.random.default_rng(0).normal(size=60)
    idata = fit_bayesian_ar(
        train,
        ar_order=1,
        draws=50,
        tune=50,
        chains=1,
        target_accept=0.9,
        cores=1,
        random_seed=0,
    )
    assert "rho" in idata.posterior
