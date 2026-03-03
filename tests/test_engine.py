"""
tests/test_engine.py
====================
End-to-end tests for XVAEngine.run().

Covers:
  - All 5 XVA scalars present and have correct types
  - Grid, EE, ENE arrays in result with expected shapes
  - CVA >= 0, DVA >= 0, FVA >= 0, MVA >= 0, KVA >= 0
  - Determinism: same seed -> identical CVA/DVA/FVA
  - Output directory created with all expected bundle artifacts
  - run_id present in result
"""

import pathlib
import polars as pl
import numpy as np
import pytest

from xva_engine.engine import XVAEngine
from xva_engine.io.schemas import ZCBTa, NettingSet


@pytest.fixture(scope="module")
def engine(
    mixed_netting_set,
    sample_curve_snapshot,
    sample_cpty_credit,
    sample_bank_credit,
    model_config,
    tmp_path_factory,
):
    out = str(tmp_path_factory.mktemp("xva_output"))
    return XVAEngine(
        netting_set=mixed_netting_set,
        curve_snapshot=sample_curve_snapshot,
        cpty_credit=sample_cpty_credit,
        bank_credit=sample_bank_credit,
        config=model_config,
        funding_spread=0.01,
        cost_of_capital=0.08,
        output_dir=out,
    )


@pytest.fixture(scope="module")
def result(engine):
    return engine.run()


class TestEngineResultKeys:
    def test_run_id_present(self, result):
        assert "run_id" in result

    def test_grid_days_present(self, result):
        assert "grid_days" in result

    def test_EE_present(self, result):
        assert "EE" in result

    def test_ENE_present(self, result):
        assert "ENE" in result

    def test_all_scalars_present(self, result):
        for key in ("CVA", "DVA", "FVA", "MVA", "KVA"):
            assert key in result

    def test_all_tables_present(self, result):
        for key in ("cva_table", "dva_table", "fva_table", "mva_table", "kva_table"):
            assert key in result

    def test_ee_by_type_present(self, result):
        assert "ee_by_type" in result

    def test_V_sample_by_type_present(self, result):
        assert "V_sample_by_type" in result


class TestEngineResultTypes:
    def test_run_id_is_str(self, result):
        assert isinstance(result["run_id"], str)

    def test_grid_days_is_ndarray(self, result):
        assert isinstance(result["grid_days"], np.ndarray)

    def test_EE_is_ndarray(self, result):
        assert isinstance(result["EE"], np.ndarray)

    def test_ENE_is_ndarray(self, result):
        assert isinstance(result["ENE"], np.ndarray)

    def test_xva_scalars_are_float(self, result):
        for key in ("CVA", "DVA", "FVA", "MVA", "KVA"):
            assert isinstance(result[key], float), f"{key} should be float"

    def test_tables_are_polars_dataframes(self, result):
        for key in ("cva_table", "dva_table", "fva_table", "mva_table", "kva_table"):
            assert isinstance(result[key], pl.DataFrame)

    def test_ee_by_type_is_dict(self, result):
        assert isinstance(result["ee_by_type"], dict)

    def test_V_sample_by_type_is_dict(self, result):
        assert isinstance(result["V_sample_by_type"], dict)

    def test_ee_by_type_values_are_arrays(self, result):
        for v in result["ee_by_type"].values():
            assert isinstance(v, np.ndarray)

    def test_V_sample_by_type_keys_match_ee_keys(self, result):
        assert set(result["V_sample_by_type"].keys()) == set(
            result["ee_by_type"].keys()
        )

    def test_ee_by_type_keys_are_trade_type_strings(self, result):
        for k in result["ee_by_type"]:
            assert isinstance(k, str)


class TestEngineArrayShapes:
    def test_EE_shape_matches_grid(self, result):
        assert result["EE"].shape == result["grid_days"].shape

    def test_ENE_shape_matches_grid(self, result):
        assert result["ENE"].shape == result["grid_days"].shape

    def test_table_row_counts_equal_grid_steps(self, result):
        n_steps = len(result["grid_days"]) - 1
        for key in ("cva_table", "dva_table", "fva_table", "mva_table", "kva_table"):
            assert len(result[key]) == n_steps, f"{key} row count mismatch"


class TestEngineSignConventions:
    def test_CVA_non_negative(self, result):
        assert result["CVA"] >= 0

    def test_DVA_non_negative(self, result):
        assert result["DVA"] >= 0

    def test_FVA_non_negative(self, result):
        assert result["FVA"] >= 0

    def test_MVA_non_negative(self, result):
        assert result["MVA"] >= 0

    def test_KVA_non_negative(self, result):
        assert result["KVA"] >= 0

    def test_EE_non_negative(self, result):
        assert np.all(result["EE"] >= 0)

    def test_ENE_non_negative(self, result):
        # ENE = mean(max(C - V, 0)) >= 0 by construction
        assert np.all(result["ENE"] >= 0)


class TestEngineDeterminism:
    def test_same_seed_gives_same_CVA(
        self,
        mixed_netting_set,
        sample_curve_snapshot,
        sample_cpty_credit,
        sample_bank_credit,
        model_config,
        tmp_path_factory,
    ):
        out = str(tmp_path_factory.mktemp("det_test"))
        kwargs = dict(
            netting_set=mixed_netting_set,
            curve_snapshot=sample_curve_snapshot,
            cpty_credit=sample_cpty_credit,
            bank_credit=sample_bank_credit,
            config=model_config,
            output_dir=out,
        )
        r1 = XVAEngine(**kwargs).run()
        r2 = XVAEngine(**kwargs).run()
        assert abs(r1["CVA"] - r2["CVA"]) < 1e-12

    def test_same_seed_gives_same_EE(
        self,
        mixed_netting_set,
        sample_curve_snapshot,
        sample_cpty_credit,
        sample_bank_credit,
        model_config,
        tmp_path_factory,
    ):
        out = str(tmp_path_factory.mktemp("det_test2"))
        kwargs = dict(
            netting_set=mixed_netting_set,
            curve_snapshot=sample_curve_snapshot,
            cpty_credit=sample_cpty_credit,
            bank_credit=sample_bank_credit,
            config=model_config,
            output_dir=out,
        )
        r1 = XVAEngine(**kwargs).run()
        r2 = XVAEngine(**kwargs).run()
        np.testing.assert_array_equal(r1["EE"], r2["EE"])


class TestEngineBundleArtifacts:
    def test_output_dir_created(self, engine, result):
        run_path = pathlib.Path(engine.output_dir) / result["run_id"]
        assert run_path.is_dir()

    def test_run_config_json_exists(self, engine, result):
        run_path = pathlib.Path(engine.output_dir) / result["run_id"]
        assert (run_path / "run_config.json").exists()

    def test_EE_npy_exists(self, engine, result):
        run_path = pathlib.Path(engine.output_dir) / result["run_id"]
        assert (run_path / "EE.npy").exists()

    def test_ENE_npy_exists(self, engine, result):
        run_path = pathlib.Path(engine.output_dir) / result["run_id"]
        assert (run_path / "ENE.npy").exists()

    def test_math_report_md_exists(self, engine, result):
        run_path = pathlib.Path(engine.output_dir) / result["run_id"]
        assert (run_path / "math_report.md").exists()

    def test_cva_parquet_exists(self, engine, result):
        run_path = pathlib.Path(engine.output_dir) / result["run_id"]
        assert (run_path / "cva_table.parquet").exists()


class TestEngineSameTypeTrades:
    """Cover the branch where a second trade of the same type accumulates
    into an already-initialised V_by_type entry (line 92 else-branch)."""

    def test_two_zcbs_produce_valid_ee_by_type(
        self,
        sample_curve_snapshot,
        sample_cpty_credit,
        sample_bank_credit,
        model_config,
        tmp_path_factory,
    ):
        ns = NettingSet(
            netting_set_id="NS-2ZCB",
            counterparty_id="CPTY-1",
            trades=[
                ZCBTa(
                    trade_id="ZCB-A",
                    netting_set_id="NS-2ZCB",
                    notional=500_000,
                    maturity_date=365,
                ),
                ZCBTa(
                    trade_id="ZCB-B",
                    netting_set_id="NS-2ZCB",
                    notional=500_000,
                    maturity_date=730,
                ),
            ],
        )
        out = str(tmp_path_factory.mktemp("two_zcb"))
        eng = XVAEngine(
            netting_set=ns,
            curve_snapshot=sample_curve_snapshot,
            cpty_credit=sample_cpty_credit,
            bank_credit=sample_bank_credit,
            config=model_config,
            output_dir=out,
        )
        rs = eng.run()
        assert "ZCB" in rs["ee_by_type"]
        assert len(rs["ee_by_type"]) == 1
        assert isinstance(rs["ee_by_type"]["ZCB"], np.ndarray)


class TestEngineWithCsaId:
    """Cover the `elif netting_set.csa_id:` fallback branch in XVAEngine.__init__."""

    def test_csa_id_fallback_sets_perfect_vm(
        self,
        zcb_trade,
        sample_curve_snapshot,
        sample_cpty_credit,
        sample_bank_credit,
        model_config,
        tmp_path_factory,
    ):
        ns = NettingSet(
            netting_set_id="NS-CSAID",
            counterparty_id="CPTY-1",
            csa_id="CSA-001",
            trades=[zcb_trade],
        )
        out = str(tmp_path_factory.mktemp("csa_id_test"))
        eng = XVAEngine(
            netting_set=ns,
            curve_snapshot=sample_curve_snapshot,
            cpty_credit=sample_cpty_credit,
            bank_credit=sample_bank_credit,
            config=model_config,
            output_dir=out,
        )
        assert eng.csa_schema is not None
        assert eng.csa_schema.mode == "perfect_vm"
