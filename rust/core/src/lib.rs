//! AR(1) path simulation: y_t = intercept + phi * y_{t-1} + sigma * z.

struct Lcg(u64);

impl Lcg {
    fn new(seed: u64) -> Self {
        Self(seed)
    }
    fn normal(&mut self) -> f64 {
        self.0 = self.0.wrapping_mul(6364136223846793005).wrapping_add(1);
        let u1 = ((self.0 >> 33) as f64 / (1u64 << 31) as f64).max(1e-12);
        self.0 = self.0.wrapping_mul(6364136223846793005).wrapping_add(1);
        let u2 = (self.0 >> 33) as f64 / (1u64 << 31) as f64;
        (-2.0 * u1.ln()).sqrt() * (2.0 * std::f64::consts::PI * u2).cos()
    }
}

/// Flattened paths: length n_paths * horizon.
pub fn simulate_ar1_paths(
    y0: f64,
    intercept: f64,
    phi: f64,
    sigma: f64,
    horizon: usize,
    n_paths: usize,
    seed: u64,
) -> Vec<f64> {
    let mut rng = Lcg::new(seed);
    let mut out = Vec::with_capacity(n_paths * horizon);
    for _ in 0..n_paths {
        let mut y = y0;
        for _ in 0..horizon {
            y = intercept + phi * y + sigma * rng.normal();
            out.push(y);
        }
    }
    out
}
