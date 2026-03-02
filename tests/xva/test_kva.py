"""
tests/xva/test_kva.py
=====================
100% coverage for src/xva_engine/xva/kva.py

Tests:
  - CapitalModel.capital() with 1D and 2D array inputs
  - compute_kva() with default (zero) capital model → KVA = 0
  - compute_kva() with custom non-zero capital model → KVA > 0
  - Scalar vs array cost_of_capital branches
  - Output type / column presence / row count
"""
import numpy as np
import pytest
import polars as pl

from xva_engine.xva.kva import CapitalModel, EECapital, compute_kva


GRID = np.array([0.0, 182.0, 365.0, 730.0, 1825.0])
N_PATHS = 50


@pytest.fixture(scope="module")
def V_paths():
    rng = np.random.default_rng(0)
    return rng.normal(0, 200_000, size=(len(GRID), N_PATHS))


class ConstantCapitalModel(CapitalModel):
    """Returns a fixed positive capital for every scenario."""
    def __init__(self, k_value: float = 100_000.0):
        self._val = k_value

    def capital(self, t_days: float, V_ns_paths: np.ndarray) -> np.ndarray:
        n = V_ns_paths.shape[1] if V_ns_paths.ndim == 2 else len(V_ns_paths)
        return np.full(n, self._val)


class TestCapitalModel:
    def test_base_capital_1d(self):
        model = CapitalModel()
        paths_1d = np.ones(15)
        result = model.capital(0.0, paths_1d)
        np.testing.assert_array_equal(result, np.zeros(15))

    def test_base_capital_2d(self):
        model = CapitalModel()
        paths_2d = np.ones((5, 15))
        result = model.capital(0.0, paths_2d)
        np.testing.assert_array_equal(result, np.zeros(15))


class TestEECapital:
    def test_positive_paths_give_positive_capital(self):
        model = EECapital(rwa_factor=0.08)
        paths = np.array([100_000.0, 200_000.0, 50_000.0])
        result = model.capital(365.0, paths)
        np.testing.assert_allclose(result, paths * 0.08)

    def test_negative_paths_give_zero_capital(self):
        """Capital floored at zero for in-the-money counterparty positions."""
        model = EECapital(rwa_factor=0.08)
        paths = np.array([-100_000.0, -50_000.0, -200.0])
        result = model.capital(365.0, paths)
        np.testing.assert_array_equal(result, np.zeros(3))

    def test_mixed_paths(self):
        model = EECapital(rwa_factor=0.10)
        paths = np.array([300_000.0, -100_000.0, 0.0])
        result = model.capital(365.0, paths)
        np.testing.assert_allclose(result, [30_000.0, 0.0, 0.0])

    def test_rwa_factor_scales_output(self):
        paths = np.full(10, 1_000_000.0)
        r1 = EECapital(rwa_factor=0.04).capital(0.0, paths)
        r2 = EECapital(rwa_factor=0.08).capital(0.0, paths)
        np.testing.assert_allclose(r2, r1 * 2.0)

    def test_compute_kva_with_ee_capital_gives_positive_kva(self, V_paths, flat_curve):
        rng = np.random.default_rng(99)
        pos_paths = np.abs(rng.normal(100_000, 50_000, size=(len(GRID), N_PATHS)))
        df = compute_kva(GRID, pos_paths, flat_curve, 0.08, capital_model=EECapital())
        assert df["kva_contribution"].sum() > 0.0


class TestComputeKVA:
    def test_returns_polars_dataframe(self, V_paths, flat_curve):
        df = compute_kva(GRID, V_paths, flat_curve, 0.08)
        assert isinstance(df, pl.DataFrame)

    def test_has_required_columns(self, V_paths, flat_curve):
        df = compute_kva(GRID, V_paths, flat_curve, 0.08)
        for col in ("t_end_days", "df", "E_K", "cost_of_capital", "dt_years", "kva_contribution"):
            assert col in df.columns

    def test_row_count_equals_grid_steps(self, V_paths, flat_curve):
        df = compute_kva(GRID, V_paths, flat_curve, 0.08)
        assert len(df) == len(GRID) - 1

    def test_default_capital_model_gives_zero_kva(self, V_paths, flat_curve):
        df = compute_kva(GRID, V_paths, flat_curve, 0.08)
        assert abs(df["kva_contribution"].sum()) < 1e-12

    def test_explicit_none_capital_model_gives_zero_kva(self, V_paths, flat_curve):
        df = compute_kva(GRID, V_paths, flat_curve, 0.08, capital_model=None)
        assert abs(df["kva_contribution"].sum()) < 1e-12

    def test_custom_capital_model_gives_positive_kva(self, V_paths, flat_curve):
        df = compute_kva(GRID, V_paths, flat_curve, 0.08, capital_model=ConstantCapitalModel(100_000))
        assert df["kva_contribution"].sum() > 0

    def test_zero_cost_gives_zero_kva(self, V_paths, flat_curve):
        df = compute_kva(GRID, V_paths, flat_curve, 0.0, capital_model=ConstantCapitalModel(100_000))
        assert abs(df["kva_contribution"].sum()) < 1e-12

    def test_scalar_cost_branch(self, V_paths, flat_curve):
        df = compute_kva(GRID, V_paths, flat_curve, 0.05, capital_model=ConstantCapitalModel())
        assert (df["cost_of_capital"] == 0.05).all()

    def test_array_cost_branch(self, V_paths, flat_curve):
        costs = np.linspace(0.05, 0.12, len(GRID))
        df = compute_kva(GRID, V_paths, flat_curve, costs, capital_model=ConstantCapitalModel())
        np.testing.assert_allclose(df["cost_of_capital"].to_numpy(), costs[1:], rtol=1e-10)

    def test_array_cost_equals_scalar_cost(self, V_paths, flat_curve):
        c = 0.07
        df_s = compute_kva(GRID, V_paths, flat_curve, c, capital_model=ConstantCapitalModel())
        df_a = compute_kva(GRID, V_paths, flat_curve, np.full(len(GRID), c), capital_model=ConstantCapitalModel())
        np.testing.assert_allclose(
            df_s["kva_contribution"].to_numpy(),
            df_a["kva_contribution"].to_numpy(),
            rtol=1e-12
        )
