"""
tests/products/test_irs.py
==========================
100% coverage for src/xva_engine/products/irs.py

Exercises IrsPricer construction, pv_deterministic() for
both payer and receiver directions, and pv_pathwise() across
all branches.
"""
import numpy as np
import pytest

from xva_engine.products.irs import IrsPricer


@pytest.fixture
def payer_pricer(irs_payer_trade):
    return IrsPricer(irs_payer_trade)


@pytest.fixture
def receiver_pricer(irs_receiver_trade):
    return IrsPricer(irs_receiver_trade)


class TestIrsPricerConstruction:
    def test_stores_trade(self, payer_pricer, irs_payer_trade):
        assert payer_pricer.trade is irs_payer_trade

    def test_schedule_ends_at_maturity(self, payer_pricer, irs_payer_trade):
        assert payer_pricer.schedule[-1] == irs_payer_trade.maturity_date

    def test_alphas_positive(self, payer_pricer):
        assert np.all(payer_pricer.alphas > 0)

    def test_alphas_sum_roughly_to_tenor_years(self, payer_pricer, irs_payer_trade):
        tenor_years = (irs_payer_trade.maturity_date - irs_payer_trade.start_date) / 365.0
        assert abs(sum(payer_pricer.alphas) - tenor_years) < 0.5  # within half a year


class TestIrsPvDeterministic:
    def test_payer_returns_float(self, payer_pricer, flat_curve):
        pv = payer_pricer.pv_deterministic(0, flat_curve)
        assert isinstance(pv, float)

    def test_receiver_opposite_sign_to_payer(self, payer_pricer, receiver_pricer, flat_curve):
        pv_pay = payer_pricer.pv_deterministic(0, flat_curve)
        pv_rec = receiver_pricer.pv_deterministic(0, flat_curve)
        assert abs(pv_pay + pv_rec) < 1.0  # payer + receiver ≈ 0

    def test_after_maturity_returns_zero(self, payer_pricer, flat_curve, irs_payer_trade):
        pv = payer_pricer.pv_deterministic(irs_payer_trade.maturity_date + 1, flat_curve)
        assert pv == 0.0

    def test_at_maturity_returns_zero(self, payer_pricer, flat_curve, irs_payer_trade):
        pv = payer_pricer.pv_deterministic(irs_payer_trade.maturity_date, flat_curve)
        assert pv == 0.0

    def test_midway_evaluation(self, payer_pricer, flat_curve):
        # Evaluation mid-life – should still return a finite float
        pv = payer_pricer.pv_deterministic(365, flat_curve)
        assert np.isfinite(pv)


class TestIrsPvPathwise:
    def test_shape_matches_paths(self, payer_pricer, flat_curve, hw_params):
        x = np.zeros(300)
        pv = payer_pricer.pv_pathwise(0, x, flat_curve, hw_params.mean_reversion, hw_params.volatility)
        assert pv.shape == (300,)

    def test_at_maturity_all_zero(self, payer_pricer, flat_curve, irs_payer_trade, hw_params):
        x = np.random.default_rng(2).standard_normal(200)
        pv = payer_pricer.pv_pathwise(
            irs_payer_trade.maturity_date, x, flat_curve,
            hw_params.mean_reversion, hw_params.volatility,
        )
        np.testing.assert_array_equal(pv, 0.0)

    def test_after_maturity_all_zero(self, payer_pricer, flat_curve, irs_payer_trade, hw_params):
        x = np.random.default_rng(3).standard_normal(200)
        pv = payer_pricer.pv_pathwise(
            irs_payer_trade.maturity_date + 60, x, flat_curve,
            hw_params.mean_reversion, hw_params.volatility,
        )
        np.testing.assert_array_equal(pv, 0.0)

    def test_payer_receiver_symmetry_at_x0(self, payer_pricer, receiver_pricer, flat_curve, hw_params):
        x = np.zeros(100)
        pv_pay = payer_pricer.pv_pathwise(0, x, flat_curve, hw_params.mean_reversion, hw_params.volatility)
        pv_rec = receiver_pricer.pv_pathwise(0, x, flat_curve, hw_params.mean_reversion, hw_params.volatility)
        np.testing.assert_allclose(pv_pay + pv_rec, 0.0, atol=1.0)

    def test_no_nan_or_inf(self, payer_pricer, flat_curve, hw_params):
        rng = np.random.default_rng(7)
        x = rng.standard_normal(500) * 0.02
        pv = payer_pricer.pv_pathwise(500, x, flat_curve, hw_params.mean_reversion, hw_params.volatility)
        assert np.all(np.isfinite(pv))

    def test_receiver_pathwise_no_nan(self, receiver_pricer, flat_curve, hw_params):
        x = np.zeros(100)
        pv = receiver_pricer.pv_pathwise(0, x, flat_curve, hw_params.mean_reversion, hw_params.volatility)
        assert np.all(np.isfinite(pv))
