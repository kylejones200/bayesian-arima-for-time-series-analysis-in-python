import json
from pathlib import Path

import pytest

from bayesian_arima_ts.config import load_config
from bayesian_arima_ts.runner import run_pipeline


@pytest.mark.slow
def test_full_pipeline_mcmc_vs_arima(tmp_path: Path):
    """End-to-end: full MCMC + pmdarima on airline passengers; writes comparison JSON."""
    cfg = load_config()
    cfg["data"] = {
        **(cfg.get("data") or {}),
        "dataset": "airline_passengers",
        "test_size": 24,
    }
    cfg["classical"] = {**(cfg.get("classical") or {}), "compare_fixed_ar": True}
    cfg["output"] = {
        **(cfg.get("output") or {}),
        "figures_dir": str(tmp_path / "figures"),
        "results_path": str(tmp_path / "results.json"),
        "save_trace": False,
    }
    # Full sampling per config.yaml (1000 tune + 1000 draws x 2 chains)
    result = run_pipeline(cfg)

    results_file = Path(result["results_path"])
    assert results_file.is_file()
    payload = json.loads(results_file.read_text(encoding="utf-8"))

    assert payload["dataset"] == "airline_passengers"
    assert payload["mcmc"]["r_hat_max"] < 1.05
    assert payload["mcmc"]["divergences"] == 0
    assert len(payload["holdout_actual"]) == payload["test_size"]
    assert len(payload["bayesian_forecast"]) == payload["test_size"]
    assert len(payload["arima_forecast"]) == payload["test_size"]
    assert "bayesian_metrics" in payload
    assert "arima_metrics" in payload
    assert "forecast_comparison" in payload
    assert result["forecast_correlation"] == pytest.approx(
        payload["forecast_comparison"]["correlation"], rel=1e-5
    )
    assert payload["bayes_vs_fixed_ar"] is not None
    assert payload["bayes_vs_fixed_ar"]["correlation"] > 0.5
    # Posterior AR(1) coefficient should be close to pmdarima MLE (sign/magnitude)
    assert abs(abs(payload["posterior"]["rho_mean"][1]) - 0.21) < 0.08
