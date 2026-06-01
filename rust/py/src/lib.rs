use bayesian_arima_for_time_series_analysis_in_python_core::simulate_ar1_paths;
use numpy::{PyArray1, IntoPyArray};
use pyo3::prelude::*;

#[pyfunction]
#[pyo3(signature = (y0, intercept, phi, sigma, horizon, n_paths, seed=42))]
fn simulate_ar1_paths_py<'py>(
    py: Python<'py>,
    y0: f64,
    intercept: f64,
    phi: f64,
    sigma: f64,
    horizon: usize,
    n_paths: usize,
    seed: u64,
) -> PyResult<Bound<'py, PyArray1<f64>>> {
    Ok(simulate_ar1_paths(y0, intercept, phi, sigma, horizon, n_paths, seed).into_pyarray(py))
}

#[pyfunction]
#[pyo3(signature = (y0, intercept, phi, sigma, horizon, n_paths, seed=42, iterations=200))]
fn bench_kernel_py(
    y0: f64,
    intercept: f64,
    phi: f64,
    sigma: f64,
    horizon: usize,
    n_paths: usize,
    seed: u64,
    iterations: usize,
) -> PyResult<f64> {
    let start = std::time::Instant::now();
    for _ in 0..iterations {
        let _ = simulate_ar1_paths(y0, intercept, phi, sigma, horizon, n_paths, seed);
    }
    Ok(start.elapsed().as_secs_f64())
}

#[pymodule]
fn bayesian_arima_for_time_series_analysis_in_python_rs(m: &Bound<'_, PyModule>) -> PyResult<()> {
    m.add_function(wrap_pyfunction!(simulate_ar1_paths_py, m)?)?;
    m.add_function(wrap_pyfunction!(bench_kernel_py, m)?)?;
    Ok(())
}
