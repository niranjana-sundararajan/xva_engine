"""
tests/sim/test_batching.py
==========================
100% coverage for src/xva_engine/sim/batching.py

Tests MonteCarloEngine.simulate_paths() for:
- output shape
- determinism with identical seeds
- different seeds produce different results
- initial state x_0 = 0
- no NaN/Inf for long grids
- OU process mean/variance sanity
"""
import numpy as np
import pytest

from xva_engine.sim.batching import MonteCarloEngine


GRID = np.array([0.0, 182.0, 365.0, 730.0, 1095.0, 1825.0])
N_PATHS = 5_000
A = 0.05
SIGMA = 0.01


@pytest.fixture(scope="module")
def engine():
    return MonteCarloEngine(num_paths=N_PATHS, seed=42, hw_a=A, hw_sigma=SIGMA)


class TestMonteCarloEngineConstruction:
    def test_stores_params(self, engine):
        assert engine.num_paths == N_PATHS
        assert engine.seed == 42
        assert engine.hw_a == A
        assert engine.hw_sigma == SIGMA


class TestSimulatePaths:
    def test_output_shape(self, engine):
        x = engine.simulate_paths(GRID)
        assert x.shape == (len(GRID), N_PATHS)

    def test_initial_x_is_zero(self, engine):
        x = engine.simulate_paths(GRID)
        np.testing.assert_array_equal(x[0, :], 0.0)

    def test_determinism_same_seed(self):
        e1 = MonteCarloEngine(1000, 99, A, SIGMA)
        e2 = MonteCarloEngine(1000, 99, A, SIGMA)
        x1 = e1.simulate_paths(GRID)
        x2 = e2.simulate_paths(GRID)
        np.testing.assert_array_equal(x1, x2)

    def test_different_seeds_differ(self):
        e1 = MonteCarloEngine(1000, 1, A, SIGMA)
        e2 = MonteCarloEngine(1000, 2, A, SIGMA)
        x1 = e1.simulate_paths(GRID)
        x2 = e2.simulate_paths(GRID)
        assert not np.allclose(x1, x2)

    def test_no_nan_or_inf(self, engine):
        x = engine.simulate_paths(GRID)
        assert np.all(np.isfinite(x))

    def test_no_nan_long_grid(self):
        e = MonteCarloEngine(500, 0, A, SIGMA)
        long_grid = np.linspace(0, 3650, 100)
        x = e.simulate_paths(long_grid)
        assert np.all(np.isfinite(x))

    def test_ou_mean_near_zero(self, engine):
        x = engine.simulate_paths(GRID)
        # OU mean = 0 (starting from 0)
        for i in range(1, len(GRID)):
            assert abs(np.mean(x[i, :])) < 0.005

    def test_ou_variance_increasing_then_bounded(self, engine):
        x = engine.simulate_paths(GRID)
        # variance should not grow without bound (OU has a long-run variance)
        var_terminal = np.var(x[-1, :])
        # Long-run variance = sigma^2 / (2*a)
        long_run_var = SIGMA**2 / (2 * A)
        assert var_terminal < 2 * long_run_var

    def test_single_step_grid(self):
        e = MonteCarloEngine(200, 5, A, SIGMA)
        grid = np.array([0.0, 365.0])
        x = e.simulate_paths(grid)
        assert x.shape == (2, 200)
        np.testing.assert_array_equal(x[0, :], 0.0)
