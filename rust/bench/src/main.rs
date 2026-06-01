use bayesian_arima_for_time_series_analysis_in_python_core::simulate_ar1_paths;

fn main() {
    for _ in 0..500 {
        let _ = simulate_ar1_paths(1.0, 0.1, 0.8, 0.05, 48, 500, 42);
    }
}
