"""
tests/xva/test_fva.py
=====================
100% coverage for src/xva_engine/xva/fva.py

Tests compute_fva() output type/columns, scalar vs array spread
branches, sign conventions, and zero-EE edge case.
"""

import numpy as np
import pytest
import polars as pl

from xva_engine.xva.fva import compute_fva


GRID = np.array([0.0, 182.0, 365.0, 730.0, 1825.0])


@pytest.fixture(scope="module")
def sample_EE():
    return np.linspace(0, 300_000, len(GRID))


@pytest.fixture(scope="module")
def sample_C():
    # Flat zero collateral (no CSA)
    return np.zeros(len(GRID))


class TestComputeFVAScalarSpread:
    """Tests when funding_spread is a scalar (exercises np.full branch)."""

    def test_returns_polars_dataframe(self, sample_EE, sample_C, flat_curve):
        df = compute_fva(GRID, sample_EE, sample_C, flat_curve, 0.01)
        assert isinstance(df, pl.DataFrame)

    def test_has_required_columns(self, sample_EE, sample_C, flat_curve):
        df = compute_fva(GRID, sample_EE, sample_C, flat_curve, 0.01)
        for col in (
            "t_end_days",
            "df",
            "EE_net",
            "funding_spread",
            "dt_years",
            "fva_contribution",
        ):
            assert col in df.columns

    def test_row_count_equals_grid_steps(self, sample_EE, sample_C, flat_curve):
        df = compute_fva(GRID, sample_EE, sample_C, flat_curve, 0.01)
        assert len(df) == len(GRID) - 1

    def test_contributions_non_negative(self, sample_EE, sample_C, flat_curve):
        df = compute_fva(GRID, sample_EE, sample_C, flat_curve, 0.01)
        assert (df["fva_contribution"] >= 0).all()

    def test_total_fva_positive(self, sample_EE, sample_C, flat_curve):
        df = compute_fva(GRID, sample_EE, sample_C, flat_curve, 0.01)
        assert df["fva_contribution"].sum() > 0

    def test_zero_EE_gives_zero_FVA(self, sample_C, flat_curve):
        EE_zero = np.zeros(len(GRID))
        df = compute_fva(GRID, EE_zero, sample_C, flat_curve, 0.01)
        assert abs(df["fva_contribution"].sum()) < 1e-12

    def test_zero_spread_gives_zero_FVA(self, sample_EE, sample_C, flat_curve):
        df = compute_fva(GRID, sample_EE, sample_C, flat_curve, 0.0)
        assert abs(df["fva_contribution"].sum()) < 1e-12

    def test_funding_spread_column_values(self, sample_EE, sample_C, flat_curve):
        spread = 0.02
        df = compute_fva(GRID, sample_EE, sample_C, flat_curve, spread)
        assert (df["funding_spread"] == spread).all()


class TestComputeFVAArraySpread:
    """Tests when funding_spread is an array (exercises np.asarray branch)."""

    def test_array_spread_accepted(self, sample_EE, sample_C, flat_curve):
        spreads = np.full(len(GRID), 0.015)
        df = compute_fva(GRID, sample_EE, sample_C, flat_curve, spreads)
        assert isinstance(df, pl.DataFrame)

    def test_array_spread_values_in_column(self, sample_EE, sample_C, flat_curve):
        spreads = np.linspace(0.005, 0.02, len(GRID))
        df = compute_fva(GRID, sample_EE, sample_C, flat_curve, spreads)
        expected = spreads[1:]
        np.testing.assert_allclose(
            df["funding_spread"].to_numpy(), expected, rtol=1e-10
        )

    def test_array_vs_equivalent_scalar_match(self, sample_EE, sample_C, flat_curve):
        spread = 0.012
        df_scalar = compute_fva(GRID, sample_EE, sample_C, flat_curve, spread)
        df_array = compute_fva(
            GRID, sample_EE, sample_C, flat_curve, np.full(len(GRID), spread)
        )
        np.testing.assert_allclose(
            df_scalar["fva_contribution"].to_numpy(),
            df_array["fva_contribution"].to_numpy(),
            rtol=1e-12,
        )

    def test_mismatched_spread_array_raises(self, sample_EE, sample_C, flat_curve):
        with pytest.raises(ValueError, match="length"):
            compute_fva(GRID, sample_EE, sample_C, flat_curve, np.array([0.01, 0.02]))
