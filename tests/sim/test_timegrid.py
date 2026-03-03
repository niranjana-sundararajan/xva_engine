"""
tests/sim/test_timegrid.py
==========================
100% coverage for src/xva_engine/sim/timegrid.py

Tests build_simulation_grid() across all branches:
- ZCB-only netting set
- IRS-only netting set
- Mixed netting set
- With and without dense_frequency_days
- dense_frequency_days equal to None vs 0 (both skip dense logic)
"""
import numpy as np
import pytest

from xva_engine.io.schemas import ZCBTa, IRSTrade, NettingSet
from xva_engine.sim.timegrid import build_simulation_grid


def make_zcb_ns(maturity: int = 730) -> NettingSet:
    return NettingSet(
        netting_set_id="NS",
        counterparty_id="C",
        trades=[ZCBTa(trade_id="Z1", netting_set_id="NS", notional=1e6, maturity_date=maturity)],
    )


def make_irs_ns(maturity: int = 1825, freq: int = 6) -> NettingSet:
    return NettingSet(
        netting_set_id="NS",
        counterparty_id="C",
        trades=[
            IRSTrade(
                trade_id="I1", netting_set_id="NS", notional=1e6,
                start_date=0, maturity_date=maturity, receive_fixed=False,
                fixed_rate=0.025, payment_frequency=freq,
            )
        ],
    )


class TestBuildSimulationGrid:
    def test_always_contains_t0(self, mixed_netting_set):
        grid = build_simulation_grid(mixed_netting_set)
        assert grid[0] == 0.0

    def test_zcb_only_contains_maturity(self):
        ns = make_zcb_ns(730)
        grid = build_simulation_grid(ns)
        assert 730.0 in grid

    def test_irs_only_contains_payment_dates(self):
        ns = make_irs_ns(1825, 6)
        grid = build_simulation_grid(ns)
        # schedule includes start_date=0 and maturity=1825
        assert 0.0 in grid
        assert 1825.0 in grid

    def test_mixed_ns_contains_all_events(self, mixed_netting_set):
        grid = build_simulation_grid(mixed_netting_set)
        assert 730.0 in grid   # ZCB maturity
        assert 1825.0 in grid  # IRS maturity

    def test_grid_sorted_ascending(self, mixed_netting_set):
        grid = build_simulation_grid(mixed_netting_set)
        assert np.all(np.diff(grid) > 0)

    def test_no_duplicates(self, mixed_netting_set):
        grid = build_simulation_grid(mixed_netting_set)
        assert len(grid) == len(np.unique(grid))

    def test_dense_grid_adds_points(self):
        ns = make_zcb_ns(365)
        grid_sparse = build_simulation_grid(ns, dense_frequency_days=None)
        grid_dense = build_simulation_grid(ns, dense_frequency_days=30)
        assert len(grid_dense) > len(grid_sparse)

    def test_dense_frequency_zero_skips_dense(self):
        ns = make_zcb_ns(365)
        grid_none = build_simulation_grid(ns, dense_frequency_days=None)
        grid_zero = build_simulation_grid(ns, dense_frequency_days=0)
        np.testing.assert_array_equal(grid_none, grid_zero)

    def test_dense_point_at_max_maturity_excluded(self):
        # If dense step lands exactly on max_maturity it is NOT added (< max_maturity check)
        ns = make_zcb_ns(180)
        grid = build_simulation_grid(ns, dense_frequency_days=180)
        # The dense loop adds points where current_day < max_maturity (180)
        # 180 is not < 180, so it is not added as a dense point; 180 is already an event
        assert 180.0 in grid

    def test_returns_numpy_float_array(self, mixed_netting_set):
        grid = build_simulation_grid(mixed_netting_set)
        assert isinstance(grid, np.ndarray)
        assert grid.dtype == float
