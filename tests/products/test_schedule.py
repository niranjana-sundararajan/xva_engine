"""
tests/products/test_schedule.py
================================
100% coverage for src/xva_engine/products/schedule.py

Tests year_fraction_act365f() and build_irs_schedule() including
edge cases such as single-payment tenors and even/uneven dividing.
"""
import numpy as np
import pytest

from xva_engine.products.schedule import year_fraction_act365f, build_irs_schedule


class TestYearFractionAct365F:
    def test_one_year(self):
        assert abs(year_fraction_act365f(0, 365) - 1.0) < 1e-12

    def test_half_year(self):
        assert abs(year_fraction_act365f(0, 182) - 182 / 365.0) < 1e-12

    def test_zero_interval(self):
        assert year_fraction_act365f(100, 100) == 0.0

    def test_scalar_float(self):
        result = year_fraction_act365f(0.0, 365.0)
        assert isinstance(float(result), float)

    def test_vectorised_numpy(self):
        d1 = np.array([0.0, 182.0])
        d2 = np.array([182.0, 365.0])
        result = year_fraction_act365f(d1, d2)
        assert result.shape == (2,)
        np.testing.assert_allclose(result, [182 / 365.0, 183 / 365.0], atol=1e-12)


class TestBuildIrsSchedule:
    def test_schedule_ends_at_maturity(self):
        sched = build_irs_schedule(0, 730, 6)
        assert sched[-1] == 730

    def test_first_date_is_one_step_from_start(self):
        sched = build_irs_schedule(0, 360, 6)
        assert sched[0] == 180  # 6 months = 6*30 days

    def test_schedule_sorted_ascending(self):
        sched = build_irs_schedule(0, 1825, 6)
        assert np.all(np.diff(sched) > 0)

    def test_annual_payments(self):
        sched = build_irs_schedule(0, 1095, 12)
        # 12*30=360 step; dates 360, 720, 1080, 1095
        assert sched[-1] == 1095
        assert sched[0] == 360

    def test_single_payment_short_tenor(self):
        # maturity <= step: loop never executes, only maturity appended
        sched = build_irs_schedule(0, 90, 6)
        assert len(sched) == 1
        assert sched[0] == 90

    def test_non_zero_start_date(self):
        sched = build_irs_schedule(30, 390, 6)
        # step=180; first = 30+180=210; 390>210 -> next would be 390 which equals maturity
        assert sched[-1] == 390
        assert sched[0] == 210

    def test_exact_fit(self):
        # step=365; start=0; maturity=730: loop adds 365, then exits; 730 appended
        sched = build_irs_schedule(0, 730, 12)  # step=360; 360 < 730 -> add 360; 720 < 730 -> add 720; 1080 >= 730 -> exit; append 730
        assert sched[-1] == 730

    def test_returns_numpy_array(self):
        sched = build_irs_schedule(0, 365, 6)
        assert isinstance(sched, np.ndarray)
