"""
tests/models/test_hw1f.py
=========================
100% coverage for src/xva_engine/models/hw1f.py

NUMBA_DISABLE_JIT=1 (set in conftest.py) makes @njit a no-op so
B_func, V_func, and simulate_xt_step run as plain Python and are
fully measurable by branch coverage.

Covers:
- B_func: t >= T (returns 0) and t < T
- V_func: t >= T (returns 0) and t < T
- hw1f_bond_price: t >= T (returns ones) and t < T
- simulate_xt_step: dt <= 0 (returns copy) and dt > 0
"""
import numpy as np
import pytest

from xva_engine.models.hw1f import B_func, V_func, hw1f_bond_price, simulate_xt_step


A = 0.05
SIGMA = 0.01


class TestBFunc:
    def test_t_equal_T_returns_zero(self):
        assert B_func(1.0, 1.0, A) == 0.0

    def test_t_greater_than_T_returns_zero(self):
        assert B_func(2.0, 1.0, A) == 0.0

    def test_t_less_than_T_positive(self):
        b = B_func(0.0, 1.0, A)
        assert b > 0.0

    def test_formula_correctness(self):
        t, T = 0.0, 5.0
        expected = (1 - np.exp(-A * (T - t))) / A
        assert abs(B_func(t, T, A) - expected) < 1e-12

    def test_increases_with_tenor(self):
        b1 = B_func(0.0, 1.0, A)
        b5 = B_func(0.0, 5.0, A)
        assert b5 > b1

    def test_small_a_does_not_overflow(self):
        b = B_func(0.0, 10.0, 1e-6)
        assert np.isfinite(b)


class TestVFunc:
    def test_t_equal_T_returns_zero(self):
        assert V_func(1.0, 1.0, A, SIGMA) == 0.0

    def test_t_greater_than_T_returns_zero(self):
        assert V_func(3.0, 2.0, A, SIGMA) == 0.0

    def test_t_less_than_T_positive(self):
        # V(t, T) > 0 only when t > 0 (has (1 - exp(-2*a*t)) factor)
        v = V_func(1.0, 5.0, A, SIGMA)
        assert v > 0.0

    def test_at_t0_is_zero(self):
        # V(0, T) = 0 because factor (1 - exp(-2*a*0)) = 0
        v = V_func(0.0, 5.0, A, SIGMA)
        assert abs(v) < 1e-12

    def test_finite_for_large_values(self):
        v = V_func(1.0, 20.0, A, SIGMA)
        assert np.isfinite(v)


class TestHw1fBondPrice:
    def test_t_at_maturity_returns_ones(self):
        x = np.array([0.0, 0.01, -0.01])
        p = hw1f_bond_price(1.0, 1.0, x, 0.975, 0.975, A, SIGMA)
        np.testing.assert_allclose(p, 1.0, atol=1e-9)

    def test_t_after_maturity_returns_ones(self):
        x = np.array([0.0, 0.005])
        p = hw1f_bond_price(2.0, 1.0, x, 0.975, 0.950, A, SIGMA)
        np.testing.assert_allclose(p, 1.0, atol=1e-9)

    def test_at_x0_matches_forward_df(self):
        # With x_t=0, P(t,T) = P(0,T)/P(0,t) * exp(-0.5*V(t,T))
        # At t=0, V=0, so P(0,T)/P(0,0)*exp(0) = P(0,T)/1 = P(0,T)
        P_0_t, P_0_T = 1.0, 0.950
        x = np.array([0.0])
        p = hw1f_bond_price(0.0, 2.0, x, P_0_t, P_0_T, A, SIGMA)
        assert abs(p[0] - P_0_T) < 1e-9

    def test_positive_x_decreases_price(self):
        P_0_t, P_0_T = 0.975, 0.950
        x_neg = np.array([-0.05])
        x_pos = np.array([0.05])
        p_neg = hw1f_bond_price(1.0, 2.0, x_neg, P_0_t, P_0_T, A, SIGMA)
        p_pos = hw1f_bond_price(1.0, 2.0, x_pos, P_0_t, P_0_T, A, SIGMA)
        assert p_neg[0] > p_pos[0]

    def test_prices_all_positive(self):
        rng = np.random.default_rng(0)
        x = rng.standard_normal(500) * 0.05
        p = hw1f_bond_price(0.5, 2.0, x, 0.988, 0.950, A, SIGMA)
        assert np.all(p > 0.0)

    def test_no_nan_or_inf(self):
        rng = np.random.default_rng(1)
        x = rng.standard_normal(1000) * 0.1
        p = hw1f_bond_price(1.0, 5.0, x, 0.975, 0.880, A, SIGMA)
        assert np.all(np.isfinite(p))


class TestSimulateXtStep:
    def test_dt_zero_returns_copy(self):
        x = np.array([0.01, -0.02, 0.005])
        result = simulate_xt_step(x, 0.0, A, SIGMA, np.zeros(3))
        np.testing.assert_array_equal(result, x)

    def test_dt_negative_returns_copy(self):
        x = np.array([0.01, 0.02])
        result = simulate_xt_step(x, -1.0, A, SIGMA, np.zeros(2))
        np.testing.assert_array_equal(result, x)

    def test_zero_noise_mean_reversion(self):
        # With Z=0, x_new = x_old * exp(-a*dt) — decay to zero
        x = np.array([0.1])
        dt = 1.0
        result = simulate_xt_step(x, dt, A, SIGMA, np.zeros(1))
        expected = x * np.exp(-A * dt)
        np.testing.assert_allclose(result, expected, rtol=1e-9)

    def test_shape_preserved(self):
        x = np.zeros(1000)
        Z = np.random.default_rng(0).standard_normal(1000)
        result = simulate_xt_step(x, 1.0 / 12, A, SIGMA, Z)
        assert result.shape == (1000,)

    def test_no_nan_or_inf(self):
        x = np.random.default_rng(2).standard_normal(2000) * 0.05
        Z = np.random.default_rng(3).standard_normal(2000)
        result = simulate_xt_step(x, 1.0 / 52, A, SIGMA, Z)
        assert np.all(np.isfinite(result))

    def test_ou_mean_is_zero(self):
        # Starting from x=0 with Z drawn from N(0,1): mean of x_new should be ~0
        rng = np.random.default_rng(42)
        x = np.zeros(50_000)
        Z = rng.standard_normal(50_000)
        result = simulate_xt_step(x, 1.0, A, SIGMA, Z)
        assert abs(np.mean(result)) < 0.001
