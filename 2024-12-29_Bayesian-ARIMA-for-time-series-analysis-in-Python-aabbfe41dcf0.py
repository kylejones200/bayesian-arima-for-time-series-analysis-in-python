# Description: Short example for Bayesian ARIMA for time series analysis in Python.



from sklearn.model_selection import TimeSeriesSplit
import arviz as az
import logging
import matplotlib.pyplot as plt
import numpy as np
import pandas as pd
import pmdarima as pmd
import pymc as pm

logger = logging.getLogger(__name__)
logging.basicConfig(
    level=logging.INFO,
    format="%(asctime)s [%(levelname)s] %(name)s: %(message)s",
)



# Simulated Time Series Data
np.random.seed(42)
n = 100
x = np.arange(n)
y = 5 + 0.5 * x + np.random.normal(0, 2, n)  # Linear trend with noise

# Plot the data
plt.figure(figsize=(10, 6))
plt.plot(x, y, label="Observed Data")
plt.xlabel("Time")
plt.ylabel("Value")
plt.legend()
plt.tight_layout()
plt.savefig("observed_data.png")
plt.show()

# Bayesian AR(1) Model
with pm.Model() as model:
    # Priors for AR(1) parameters
    phi = pm.Normal("phi", mu=0, sigma=1)  # AR coefficient
    sigma = pm.HalfNormal("sigma", sigma=1)  # Noise term

    # Initial value for GaussianRandomWalk
    init_dist = pm.Normal.dist(0, 10)
    y_obs = pm.GaussianRandomWalk("y_obs", sigma=sigma, init_dist=init_dist, shape=n)

    # Likelihood
    y_like = pm.Normal("y_like", mu=y_obs, sigma=sigma, observed=y)

    # Sample the posterior
    trace = pm.sample(1000, tune=1000, return_inferencedata=True, cores=1)

# Plot Results
az.plot_trace(trace)
plt.tight_layout()
plt.savefig("bayesian.png")
plt.show()

# Posterior Predictive Sampling
with model:
    posterior_predictive = pm.sample_posterior_predictive(trace)

# Inspect the shape of posterior_samples
logger.info(f"Original shape of posterior_samples: {posterior_samples.shape}")

# Ensure correct shape: (num_samples, num_time_points)
posterior_samples = posterior_samples.squeeze()  # Remove any singleton dimensions
logger.info(f"Shape after squeeze: {posterior_samples.shape}")

# Check dimensionality; if still (1000, 1, 100), reshape or index
if posterior_samples.ndim > 2:
    posterior_samples = posterior_samples[:, 0, :]  # Remove the redundant dimension
    logger.info(f"Shape after indexing: {posterior_samples.shape}")

# Compute posterior mean and credible intervals
posterior_mean = posterior_samples.mean(axis=0)  # Reduce to (100,)
lower_bound = np.percentile(posterior_samples, 2.5, axis=0)  # Reduce to (100,)
upper_bound = np.percentile(posterior_samples, 97.5, axis=0)  # Reduce to (100,)

# Validate shapes before plotting
logger.info(f"x shape: {x.shape}, posterior_mean shape: {posterior_mean.shape}")

# Plot the results
plt.figure(figsize=(10, 6))
plt.plot(x, y, label="Observed Data")
plt.plot(x, posterior_mean, label="Posterior Mean", color='r')
plt.fill_between(x, lower_bound, upper_bound, color='r', alpha=0.3, label="95% Credible Interval")
plt.xlabel("Time")
plt.ylabel("Value")
plt.legend()
plt.tight_layout()
plt.savefig("observed_predicted.png")
plt.show()

# Compute MAPE
mape = np.mean(np.abs((y - posterior_mean) / y)) * 100

logger.info(f"Mean Absolute Percentage Error (MAPE): {mape:.2f}%")

Mean Absolute Percentage Error (MAPE): 7.06%


# Train-Test Split using time series cross-validation (last fold as test)
tscv = TimeSeriesSplit(n_splits=5)
idx = np.arange(len(y))
train_idx, test_idx = list(tscv.split(idx))[ -1 ]
train, test = y[train_idx], y[test_idx]

arima = pmd.auto_arima(
    train,
    error_action="ignore",
    trace=True,
    suppress_warnings=True,
    seasonal=False,
    maxiter=5
)

forecasts = arima.predict(n_periods=test.shape[0])

# Calculate MAPE
mape = np.mean(np.abs((test - forecasts) / test)) * 100
logger.info(f"Mean Absolute Percentage Error (MAPE): {mape:.2f}%")

# Plot actual test vs. forecasts
x_test = np.arange(test.shape[0])
plt.figure(figsize=(10, 6))
plt.plot(x_test, test, color="red", label="Actual")
plt.plot(x_test, forecasts, color="blue", label="Forecast")
plt.title("Actual Test Samples vs. Forecasts")
plt.xlabel("Time Index")
plt.ylabel("Value")
plt.legend()
plt.tight_layout()
plt.show()

Total fit time: 0.101 seconds
Mean Absolute Percentage Error (MAPE): 1.81%
