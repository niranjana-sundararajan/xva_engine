"""
tests/products/test_zcb.py
==========================
100% coverage for src/xva_engine/products/zcb.py

Tests ZcbPricer.pv_deterministic() and pv_pathwise() across
all branches: before maturity, exactly at maturity, after maturity.
"""
import numpy as np
import pytest

from xva_engine.io.schemas import ZCBTa
from xva_engine.products.zcb import ZcbPricer


@pytest.fixture
def pricer(zcb_trade):
    return ZcbPricer(zcb_trade)


class TestZcbPricerConstruction:
    def test_stores_trade(self, pricer, zcb_trade):
        assert pricer.trade is zcb_trade


class TestZcbPvDeterministic:
    def test_at_t0_equals_notional_times_df(self, pricer, flat_curve, zcb_trade):
        expected = zcb_trade.notional * flat_curve.df(zcb_trade.maturity_date)
        pv = pricer.pv_deterministic(0, flat_curve)
        assert abs(pv - expected) < 1.0

    def test_at_some_t_before_maturity(self, pricer, flat_curve, zcb_trade):
        t = 365
        dfs_t = flat_curve.df(t)
        dfs_T = flat_curve.df(zcb_trade.maturity_date)
        expected = zcb_trade.notional * dfs_T / dfs_t
        pv = pricer.pv_deterministic(t, flat_curve)
        assert abs(pv - expected) < 1.0

    def test_at_maturity_returns_zero(self, pricer, flat_curve, zcb_trade):
        pv = pricer.pv_deterministic(zcb_trade.maturity_date, flat_curve)
        assert pv == 0.0

    def test_after_maturity_returns_zero(self, pricer, flat_curve, zcb_trade):
        pv = pricer.pv_deterministic(zcb_trade.maturity_date + 1, flat_curve)
        assert pv == 0.0

    def test_pv_positive_before_maturity(self, pricer, flat_curve):
        assert pricer.pv_deterministic(0, flat_curve) > 0.0


class TestZcbPvPathwise:
    def test_shape_matches_paths(self, pricer, flat_curve, hw_params):
        x = np.zeros(500)
        pv = pricer.pv_pathwise(0, x, flat_curve, hw_params.mean_reversion, hw_params.volatility)
        assert pv.shape == (500,)

    def test_at_maturity_all_zero(self, pricer, flat_curve, zcb_trade, hw_params):
        x = np.random.default_rng(0).standard_normal(200)
        pv = pricer.pv_pathwise(zcb_trade.maturity_date, x, flat_curve, hw_params.mean_reversion, hw_params.volatility)
        np.testing.assert_array_equal(pv, 0.0)

    def test_after_maturity_all_zero(self, pricer, flat_curve, zcb_trade, hw_params):
        x = np.random.default_rng(1).standard_normal(200)
        pv = pricer.pv_pathwise(zcb_trade.maturity_date + 30, x, flat_curve, hw_params.mean_reversion, hw_params.volatility)
        np.testing.assert_array_equal(pv, 0.0)

    def test_mean_x0_matches_deterministic(self, pricer, flat_curve, zcb_trade, hw_params):
        # With x_t=0 everywhere, pathwise should match deterministic
        x = np.zeros(1000)
        pv_path = pricer.pv_pathwise(0, x, flat_curve, hw_params.mean_reversion, hw_params.volatility)
        pv_det = pricer.pv_deterministic(0, flat_curve)
        np.testing.assert_allclose(pv_path, pv_det, rtol=1e-6)

    def test_no_nan_or_inf_in_paths(self, pricer, flat_curve, hw_params):
        rng = np.random.default_rng(42)
        x = rng.standard_normal(1000) * 0.05
        pv = pricer.pv_pathwise(365, x, flat_curve, hw_params.mean_reversion, hw_params.volatility)
        assert np.all(np.isfinite(pv))
