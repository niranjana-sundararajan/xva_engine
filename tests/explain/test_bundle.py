"""
tests/explain/test_bundle.py
============================
100% coverage for src/xva_engine/explain/bundle.py

Tests every method of ExplainabilityBundle:
  - __init__: directory created
  - save_config: JSON file written and parseable
  - save_array: .npy file written and loadable
  - save_table: .parquet file written and loadable
  - add_section: sections list grows
  - write_math_report: markdown file contains run_id and sections
"""
import json
import pathlib
import numpy as np
import polars as pl
import pytest

from xva_engine.explain.bundle import ExplainabilityBundle


@pytest.fixture()
def bundle(tmp_path):
    return ExplainabilityBundle(output_dir=str(tmp_path), run_id="test_run")


class TestExplainabilityBundleInit:
    def test_output_dir_created(self, tmp_path):
        b = ExplainabilityBundle(output_dir=str(tmp_path), run_id="my_run")
        assert (tmp_path / "my_run").is_dir()

    def test_run_id_stored(self, bundle):
        assert bundle.run_id == "test_run"

    def test_sections_empty_on_init(self, bundle):
        assert bundle.sections == []


class TestSaveConfig:
    def test_creates_json_file(self, bundle):
        bundle.save_config({"key": "value", "number": 42})
        path = bundle.output_dir / "run_config.json"
        assert path.exists()

    def test_json_content_matches(self, bundle):
        cfg = {"alpha": 1.0, "beta": [1, 2, 3]}
        bundle.save_config(cfg)
        path = bundle.output_dir / "run_config.json"
        loaded = json.loads(path.read_text())
        assert loaded == cfg


class TestSaveArray:
    def test_creates_npy_file(self, bundle):
        arr = np.array([1.0, 2.0, 3.0])
        bundle.save_array("my_array", arr)
        assert (bundle.output_dir / "my_array.npy").exists()

    def test_loaded_array_matches(self, bundle):
        arr = np.linspace(0, 1, 20)
        bundle.save_array("test_arr", arr)
        loaded = np.load(bundle.output_dir / "test_arr.npy")
        np.testing.assert_array_equal(arr, loaded)

    def test_2d_array_roundtrip(self, bundle):
        arr = np.ones((5, 10))
        bundle.save_array("arr_2d", arr)
        loaded = np.load(bundle.output_dir / "arr_2d.npy")
        np.testing.assert_array_equal(arr, loaded)


class TestSaveTable:
    def test_creates_parquet_file(self, bundle):
        df = pl.DataFrame({"a": [1, 2, 3], "b": [4.0, 5.0, 6.0]})
        bundle.save_table("my_table", df)
        assert (bundle.output_dir / "my_table.parquet").exists()

    def test_loaded_table_matches(self, bundle):
        df = pl.DataFrame({"x": [10, 20], "y": [0.5, 1.5]})
        bundle.save_table("tbl", df)
        loaded = pl.read_parquet(bundle.output_dir / "tbl.parquet")
        assert loaded.equals(df)


class TestAddSection:
    def test_single_section(self, bundle):
        bundle.add_section("CVA", "CVA formula explanation")
        assert len(bundle.sections) == 1

    def test_section_content(self, bundle):
        b = ExplainabilityBundle(output_dir=bundle.output_dir.parent, run_id="sec_run")
        b.add_section("DVA", "DVA explanation here")
        assert "DVA" in b.sections[0]
        assert "DVA explanation here" in b.sections[0]

    def test_multiple_sections_accumulate(self, bundle):
        b = ExplainabilityBundle(output_dir=bundle.output_dir.parent, run_id="multi_run")
        b.add_section("A", "content A")
        b.add_section("B", "content B")
        b.add_section("C", "content C")
        assert len(b.sections) == 3


class TestWriteMathReport:
    def test_creates_math_report_md(self, bundle):
        bundle.write_math_report()
        assert (bundle.output_dir / "math_report.md").exists()

    def test_report_contains_run_id(self, bundle):
        b = ExplainabilityBundle(output_dir=bundle.output_dir.parent, run_id="report_run")
        b.write_math_report()
        content = (b.output_dir / "math_report.md").read_text()
        assert "report_run" in content

    def test_report_contains_sections(self, bundle):
        b = ExplainabilityBundle(output_dir=bundle.output_dir.parent, run_id="sec_report_run")
        b.add_section("FVA", "FVA is the funding value adjustment.")
        b.write_math_report()
        content = (b.output_dir / "math_report.md").read_text()
        assert "FVA" in content
        assert "FVA is the funding value adjustment." in content

    def test_report_empty_sections_still_writes(self, bundle):
        b = ExplainabilityBundle(output_dir=bundle.output_dir.parent, run_id="empty_sec_run")
        b.write_math_report()
        content = (b.output_dir / "math_report.md").read_text()
        assert "XVA Engine Math Report" in content
