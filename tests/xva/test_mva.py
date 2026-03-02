"""
tests/xva/test_mva.py
=====================
100% coverage for src/xva_engine/xva/mva.py

Tests:
  - InitialMarginModel.im() with 1D and 2D array inputs
  - compute_mva() with default (zero) IM model → MVA = 0
  - compute_mva() with custom non-zero IM model → MVA > 0
  - Scalar vs array funding_spread branches
  - Output type / column presence / row count
"""
import numpy as np
import pytest
import polars as pl

from xva_engine.xva.mva import InitialMarginModel, PercentileIM, compute_mva


GRID = np.array([0.0, 182.0, 365.0, 730.0, 1825.0])
N_PATHS = 50


@pytest.fixture(scope="module")
def V_paths():
    rng = np.random.default_rng(42)
    return rng.normal(0, 100_000, size=(len(GRID), N_PATHS))


class ConstantIMModel(InitialMarginModel):
    """Returns a fixed positive IM for every scenario."""
    def __init__(self, im_value: float = 50_000.0):
        self._val = im_value

    def im(self, t_days: float, V_ns_paths: np.ndarray) -> np.ndarray:
        n = V_ns_paths.shape[1] if V_ns_paths.ndim == 2 else len(V_ns_paths)
        return np.full(n, self._val)


class TestInitialMarginModel:
    def test_base_im_1d(self):
        model = InitialMarginModel()
        paths_1d = np.ones(20)
        result = model.im(0.0, paths_1d)
        np.testing.assert_array_equal(result, np.zeros(20))

    def test_base_im_2d(self):
        model = InitialMarginModel()
        paths_2d = np.ones((5, 20))
        result = model.im(0.0, paths_2d)
        np.testing.assert_array_equal(result, np.zeros(20))


class TestPercentileIM:
    def test_returns_array_of_length_n_paths(self):
        model = PercentileIM()
        paths = np.random.default_rng(0).normal(0, 100_000, 200)
        result = model.im(365.0, paths)
        assert result.shape == (200,)

    def test_all_elements_equal(self):
        """PercentileIM applies a uniform scalar IM across all paths."""
        model = PercentileIM()
        paths = np.random.default_rng(1).normal(50_000, 80_000, 200)
        result = model.im(365.0, paths)
        assert np.all(result == result[0])

    def test_positive_im_when_paths_have_variance(self):
        model = PercentileIM(confidence=0.99, mpor_days=10)
        paths = np.random.default_rng(2).normal(0, 200_000, 500)
        result = model.im(365.0, paths)
        assert result[0] > 0.0

    def test_zero_im_when_paths_are_constant(self):
        """Std = 0 => IM = 0."""
        model = PercentileIM()
        paths = np.full(100, 500_000.0)
        result = model.im(365.0, paths)
        np.testing.assert_array_equal(result, np.zeros(100))

    def test_higher_confidence_gives_higher_im(self):
        rng = np.random.default_rng(3)
        paths = rng.normal(0, 100_000, 300)
        im_low  = PercentileIM(confidence=0.95).im(365.0, paths)[0]
        im_high = PercentileIM(confidence=0.99).im(365.0, paths)[0]
        assert im_high > im_low

    def test_longer_mpor_gives_higher_im(self):
        rng = np.random.default_rng(4)
        paths = rng.normal(0, 100_000, 300)
        im_short = PercentileIM(mpor_days=5).im(365.0, paths)[0]
        im_long  = PercentileIM(mpor_days=20).im(365.0, paths)[0]
        assert im_long > im_short

    def test_compute_mva_with_percentile_im_gives_positive_mva(self, V_paths, flat_curve):
        df = compute_mva(GRID, V_paths, flat_curve, 0.01, im_model=PercentileIM())
        assert df["mva_contribution"].sum() > 0.0


class TestComputeMVA:
    def test_returns_polars_dataframe(self, V_paths, flat_curve):
        df = compute_mva(GRID, V_paths, flat_curve, 0.01)
        assert isinstance(df, pl.DataFrame)

    def test_has_required_columns(self, V_paths, flat_curve):
        df = compute_mva(GRID, V_paths, flat_curve, 0.01)
        for col in ("t_end_days", "df", "E_IM", "funding_spread", "dt_years", "mva_contribution"):
            assert col in df.columns

    def test_row_count_equals_grid_steps(self, V_paths, flat_curve):
        df = compute_mva(GRID, V_paths, flat_curve, 0.01)
        assert len(df) == len(GRID) - 1

    def test_default_im_model_gives_zero_mva(self, V_paths, flat_curve):
        df = compute_mva(GRID, V_paths, flat_curve, 0.01)
        assert abs(df["mva_contribution"].sum()) < 1e-12

    def test_explicit_none_im_model_gives_zero_mva(self, V_paths, flat_curve):
        df = compute_mva(GRID, V_paths, flat_curve, 0.01, im_model=None)
        assert abs(df["mva_contribution"].sum()) < 1e-12

    def test_custom_im_model_gives_positive_mva(self, V_paths, flat_curve):
        df = compute_mva(GRID, V_paths, flat_curve, 0.01, im_model=ConstantIMModel(50_000))
        assert df["mva_contribution"].sum() > 0

    def test_zero_spread_gives_zero_mva_even_nonzero_im(self, V_paths, flat_curve):
        df = compute_mva(GRID, V_paths, flat_curve, 0.0, im_model=ConstantIMModel(50_000))
        assert abs(df["mva_contribution"].sum()) < 1e-12

    def test_scalar_spread_branch(self, V_paths, flat_curve):
        df = compute_mva(GRID, V_paths, flat_curve, 0.005, im_model=ConstantIMModel())
        assert (df["funding_spread"] == 0.005).all()

    def test_array_spread_branch(self, V_paths, flat_curve):
        spreads = np.linspace(0.005, 0.02, len(GRID))
        df = compute_mva(GRID, V_paths, flat_curve, spreads, im_model=ConstantIMModel())
        np.testing.assert_allclose(df["funding_spread"].to_numpy(), spreads[1:], rtol=1e-10)

    def test_array_spread_equals_scalar_spread(self, V_paths, flat_curve):
        s = 0.012
        df_s = compute_mva(GRID, V_paths, flat_curve, s, im_model=ConstantIMModel())
        df_a = compute_mva(GRID, V_paths, flat_curve, np.full(len(GRID), s), im_model=ConstantIMModel())
        np.testing.assert_allclose(
            df_s["mva_contribution"].to_numpy(),
            df_a["mva_contribution"].to_numpy(),
            rtol=1e-12
        )
