"""
tests/market/test_credit.py
===========================
100% coverage for src/xva_engine/market/credit.py

Tests CreditCurveModel construction, survival_prob() interpolation,
marginal_pd() sign/magnitude, and vectorised array paths.
"""
import numpy as np
import pytest

from xva_engine.io.schemas import CreditCurve, CreditPoint
from xva_engine.market.credit import CreditCurveModel


def make_credit(entity_id: str, recovery: float, points: list[tuple[int, float]]) -> CreditCurveModel:
    return CreditCurveModel(
        CreditCurve(
            entity_id=entity_id,
            recovery_rate=recovery,
            points=[CreditPoint(tenor=t, survival_prob=s) for t, s in points],
        )
    )


class TestCreditCurveModelConstruction:
    def test_stores_entity_id(self, cpty_credit_model):
        assert cpty_credit_model.entity_id == "CPTY-1"

    def test_stores_recovery_rate(self, cpty_credit_model):
        assert abs(cpty_credit_model.recovery_rate - 0.40) < 1e-12

    def test_sorts_pillars(self):
        m = make_credit("E", 0.4, [(730, 0.96), (0, 1.0), (365, 0.98)])
        assert list(m.days) == [0.0, 365.0, 730.0]

    def test_log_survival_finite(self, cpty_credit_model):
        assert np.all(np.isfinite(cpty_credit_model.log_survival))

    def test_near_zero_survival_clamped(self):
        # survival_prob = 0 should not produce -inf in log_survival
        m = make_credit("E", 0.4, [(0, 1.0), (365, 0.0)])
        assert np.all(np.isfinite(m.log_survival))


class TestSurvivalProb:
    def test_at_t0_is_one(self, cpty_credit_model):
        assert abs(cpty_credit_model.survival_prob(0) - 1.0) < 1e-9

    def test_at_pillar(self, cpty_credit_model):
        assert abs(cpty_credit_model.survival_prob(365) - 0.98) < 1e-6

    def test_between_pillars_interpolated(self, cpty_credit_model):
        s = cpty_credit_model.survival_prob(182)
        # Should be strictly between 1.0 and 0.98
        assert 0.98 < s < 1.0

    def test_in_unit_interval(self, cpty_credit_model):
        for t in [0, 100, 365, 600, 730, 1000, 1825]:
            s = cpty_credit_model.survival_prob(t)
            assert 0.0 <= s <= 1.0

    def test_non_increasing(self, cpty_credit_model):
        tenors = np.linspace(0, 1825, 100)
        s = cpty_credit_model.survival_prob(tenors)
        assert np.all(np.diff(s) <= 1e-12)

    def test_vectorised_array(self, cpty_credit_model):
        t = np.array([0.0, 365.0, 730.0, 1825.0])
        s = cpty_credit_model.survival_prob(t)
        assert s.shape == (4,)
        assert np.all(np.isfinite(s))


class TestMarginalPD:
    def test_positive_over_interval(self, cpty_credit_model):
        dpd = cpty_credit_model.marginal_pd(0, 365)
        assert dpd > 0.0

    def test_sums_to_total_mortality(self, cpty_credit_model):
        # sum of buckets ≈ 1 - S(T)
        t_grid = np.linspace(0, 1825, 20)
        total = sum(
            cpty_credit_model.marginal_pd(t_grid[i], t_grid[i + 1])
            for i in range(len(t_grid) - 1)
        )
        expected = 1.0 - cpty_credit_model.survival_prob(1825)
        assert abs(total - expected) < 1e-9

    def test_zero_over_zero_interval(self, cpty_credit_model):
        assert abs(cpty_credit_model.marginal_pd(365, 365)) < 1e-12

    def test_vectorised(self, cpty_credit_model):
        starts = np.array([0.0, 365.0, 730.0])
        ends = np.array([365.0, 730.0, 1825.0])
        dpd = cpty_credit_model.marginal_pd(starts, ends)
        assert dpd.shape == (3,)
        assert np.all(dpd >= 0.0)
