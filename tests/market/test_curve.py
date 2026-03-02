"""
tests/market/test_curve.py
==========================
100% coverage for src/xva_engine/market/curve.py

Exercises DiscountCurve construction, df() interpolation at
known pillar points and mid-points, extrapolation behaviour,
zero_rate() correctness, and array-vectorised calls.
"""
import numpy as np
import pytest

from xva_engine.io.schemas import CurveSnapshot, CurvePoint
from xva_engine.market.curve import DiscountCurve


# ---------------------------------------------------------------------------
# Helpers
# ---------------------------------------------------------------------------

def make_curve(points: list[tuple[int, float]]) -> DiscountCurve:
    return DiscountCurve(
        CurveSnapshot(
            currency="USD",
            points=[CurvePoint(tenor=t, discount_factor=df) for t, df in points],
        )
    )


# ---------------------------------------------------------------------------
# Construction
# ---------------------------------------------------------------------------

class TestDiscountCurveConstruction:
    def test_stores_currency(self, flat_curve):
        assert flat_curve.currency == "USD"

    def test_sorts_pillars(self):
        # Pillars provided out-of-order – should be sorted internally
        curve = make_curve([(730, 0.950), (0, 1.0), (365, 0.975)])
        assert list(curve.days) == [0.0, 365.0, 730.0]

    def test_log_dfs_computed(self, flat_curve):
        assert np.all(np.isfinite(flat_curve.log_dfs))


# ---------------------------------------------------------------------------
# df()
# ---------------------------------------------------------------------------

class TestDiscountCurveDf:
    def test_df_at_t0_is_one(self, flat_curve):
        assert abs(flat_curve.df(0) - 1.0) < 1e-12

    def test_df_at_known_pillar(self, flat_curve):
        assert abs(flat_curve.df(730) - 0.950) < 1e-9

    def test_df_at_another_pillar(self, flat_curve):
        assert abs(flat_curve.df(365) - 0.975) < 1e-9

    def test_df_strictly_positive_at_all_points(self, flat_curve):
        for t in [0, 90, 182, 365, 547, 730, 1095, 1825]:
            assert flat_curve.df(t) > 0.0

    def test_df_monotone_decreasing_with_time(self, flat_curve):
        tenors = np.linspace(1, 1825, 50)
        dfs = flat_curve.df(tenors)
        assert np.all(np.diff(dfs) < 0)

    def test_df_vectorised_array_input(self, flat_curve):
        t = np.array([0.0, 182.0, 365.0, 730.0])
        result = flat_curve.df(t)
        assert result.shape == (4,)
        np.testing.assert_allclose(result[0], 1.0, atol=1e-12)

    def test_df_extrapolation_left(self):
        # Requesting t below the first pillar returns the leftmost log DF
        curve = make_curve([(30, 0.998), (365, 0.975)])
        val = curve.df(0)
        # np.interp clamps to left boundary
        assert val > 0.0

    def test_df_extrapolation_right(self, flat_curve):
        # Beyond last pillar (1825) – np.interp clamps flat
        val_at_boundary = flat_curve.df(1825)
        val_beyond = flat_curve.df(2190)
        # Flat extrapolation means same log DF -> same DF
        assert abs(val_beyond - val_at_boundary) < 1e-9


# ---------------------------------------------------------------------------
# zero_rate()
# ---------------------------------------------------------------------------

class TestDiscountCurveZeroRate:
    def test_zero_rate_positive(self, flat_curve):
        for t in [182, 365, 730, 1825]:
            assert flat_curve.zero_rate(t) > 0.0

    def test_zero_rate_near_t0_no_div_by_zero(self, flat_curve):
        # t=0 uses clamp to 1e-6 years – should not raise
        r = flat_curve.zero_rate(0)
        assert np.isfinite(r)

    def test_zero_rate_consistency_with_df(self, flat_curve):
        t = 365.0
        r = flat_curve.zero_rate(t)
        implied_df = np.exp(-r * (t / 365.0))
        assert abs(implied_df - flat_curve.df(t)) < 1e-9

    def test_zero_rate_vectorised(self, flat_curve):
        t = np.array([365.0, 730.0, 1825.0])
        r = flat_curve.zero_rate(t)
        assert r.shape == (3,)
        assert np.all(np.isfinite(r))
