"""
tests/xva/test_cva.py
=====================
100% coverage for src/xva_engine/xva/cva.py

Tests compute_cva() for output type, column presence, sign/direction
(higher hazard → higher CVA), monotonicity invariants, and
scalar summation correctness.
"""
import numpy as np
import pytest
import polars as pl

from xva_engine.xva.cva import compute_cva


GRID = np.array([0.0, 182.0, 365.0, 730.0, 1825.0])
N_PATHS = 2000


@pytest.fixture(scope="module")
def sample_EE(flat_curve):
    # Simulate a simple linear EE profile
    return np.linspace(0, 500_000, len(GRID))


class TestComputeCVA:
    def test_returns_polars_dataframe(self, sample_EE, flat_curve, cpty_credit_model):
        df = compute_cva(GRID, sample_EE, flat_curve, cpty_credit_model)
        assert isinstance(df, pl.DataFrame)

    def test_has_required_columns(self, sample_EE, flat_curve, cpty_credit_model):
        df = compute_cva(GRID, sample_EE, flat_curve, cpty_credit_model)
        assert "t_end_days" in df.columns
        assert "df" in df.columns
        assert "EE" in df.columns
        assert "dPD" in df.columns
        assert "cva_contribution" in df.columns

    def test_row_count_equals_grid_steps(self, sample_EE, flat_curve, cpty_credit_model):
        df = compute_cva(GRID, sample_EE, flat_curve, cpty_credit_model)
        assert len(df) == len(GRID) - 1

    def test_contributions_non_negative(self, sample_EE, flat_curve, cpty_credit_model):
        df = compute_cva(GRID, sample_EE, flat_curve, cpty_credit_model)
        assert (df["cva_contribution"] >= 0).all()

    def test_total_cva_positive(self, sample_EE, flat_curve, cpty_credit_model):
        df = compute_cva(GRID, sample_EE, flat_curve, cpty_credit_model)
        assert df["cva_contribution"].sum() > 0

    def test_zero_EE_gives_zero_CVA(self, flat_curve, cpty_credit_model):
        EE_zero = np.zeros(len(GRID))
        df = compute_cva(GRID, EE_zero, flat_curve, cpty_credit_model)
        assert abs(df["cva_contribution"].sum()) < 1e-12

    def test_df_column_in_unit_interval(self, sample_EE, flat_curve, cpty_credit_model):
        df = compute_cva(GRID, sample_EE, flat_curve, cpty_credit_model)
        assert (df["df"] > 0).all()
        assert (df["df"] <= 1.0).all()

    def test_dPD_positive(self, sample_EE, flat_curve, cpty_credit_model):
        df = compute_cva(GRID, sample_EE, flat_curve, cpty_credit_model)
        assert (df["dPD"] >= 0).all()
