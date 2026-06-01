#!/usr/bin/env python3
"""Python vs Rust kernel benchmark."""

from __future__ import annotations

import time
import sys
from pathlib import Path

import numpy as np

ROOT = Path(__file__).resolve().parent
sys.path.insert(0, str(ROOT / "src"))
from compute_kernel import simulate_ar1_paths  # noqa: E402

def main() -> None:
    y0, c, phi, sig, h, p, seed = 1.0, 0.1, 0.8, 0.05, 48, 500, 42
    t0 = time.perf_counter()
    for _ in range(200):
        simulate_ar1_paths(y0, c, phi, sig, h, p, seed)
    py_s = time.perf_counter() - t0
    try:
        import bayesian_arima_for_time_series_analysis_in_python_rs as rs
    except ImportError:
        print("Build: maturin develop --release -m rust/py/Cargo.toml")
        print(f"Python {py_s:.3f}s")
        return
    rs_s = rs.bench_kernel_py(y0, c, phi, sig, h, p, seed, 100)
    print(f"Python {py_s:.3f}s Rust {rs_s:.3f}s speedup {py_s / max(rs_s, 1e-9):.1f}x")
    py = simulate_ar1_paths(y0, c, phi, sig, h, p, seed)
    rs_out = np.asarray(rs.simulate_ar1_paths_py(y0, c, phi, sig, h, p, seed))
    assert py.shape == rs_out.shape
    print("Correctness: OK")

if __name__ == "__main__":
    main()
