"""
tests/xva/test_dva.py
=====================
100% coverage for src/xva_engine/xva/dva.py

Tests compute_dva() for output type, columns, sign conventions
(DVA uses ENE which is negative exposure), and zero-ENE edge case.
"""
import numpy as np
import pytest
import polars as pl

from xva_engine.xva.dva import compute_dva


GRID = np.array([0.0, 182.0, 365.0, 730.0, 1825.0])


@pytest.fixture(scope="module")
def sample_ENE():
    return np.linspace(0, -200_000, len(GRID))


class TestComputeDVA:
    def test_returns_polars_dataframe(self, sample_ENE, flat_curve, bank_credit_model):
        df = compute_dva(GRID, sample_ENE, flat_curve, bank_credit_model)
        assert isinstance(df, pl.DataFrame)

    def test_has_required_columns(self, sample_ENE, flat_curve, bank_credit_model):
        df = compute_dva(GRID, sample_ENE, flat_curve, bank_credit_model)
        for col in ("t_end_days", "df", "ENE", "dPD_b", "dva_contribution"):
            assert col in df.columns

    def test_row_count_equals_grid_steps(self, sample_ENE, flat_curve, bank_credit_model):
        df = compute_dva(GRID, sample_ENE, flat_curve, bank_credit_model)
        assert len(df) == len(GRID) - 1

    def test_dva_contributions_non_negative(self, sample_ENE, flat_curve, bank_credit_model):
        # DVA is a benefit (+), but individual contributions should be non-negative
        df = compute_dva(GRID, sample_ENE, flat_curve, bank_credit_model)
        assert (df["dva_contribution"] >= 0).all()

    def test_total_dva_positive(self, sample_ENE, flat_curve, bank_credit_model):
        df = compute_dva(GRID, sample_ENE, flat_curve, bank_credit_model)
        assert df["dva_contribution"].sum() > 0

    def test_zero_ENE_gives_zero_DVA(self, flat_curve, bank_credit_model):
        ENE_zero = np.zeros(len(GRID))
        df = compute_dva(GRID, ENE_zero, flat_curve, bank_credit_model)
        assert abs(df["dva_contribution"].sum()) < 1e-12

    def test_df_column_in_unit_interval(self, sample_ENE, flat_curve, bank_credit_model):
        df = compute_dva(GRID, sample_ENE, flat_curve, bank_credit_model)
        assert (df["df"] > 0).all()
        assert (df["df"] <= 1.0).all()

    def test_dPD_positive(self, sample_ENE, flat_curve, bank_credit_model):
        df = compute_dva(GRID, sample_ENE, flat_curve, bank_credit_model)
        assert (df["dPD_b"] >= 0).all()
