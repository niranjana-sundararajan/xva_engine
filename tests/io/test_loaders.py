"""
tests/io/test_loaders.py
========================
Branch-complete tests for src/xva_engine/io/loaders.py.

Covers every branch introduced by load_portfolio_csv, load_discount_curve_json,
load_credit_curves_json, and load_model_config_json.
"""
import json
import textwrap
import pytest

from xva_engine.io.loaders import (
    load_portfolio_csv,
    load_discount_curve_json,
    load_credit_curves_json,
    load_model_config_json,
)
from xva_engine.io.schemas import (
    ZCBTa,
    IRSTrade,
    NettingSet,
    CurveSnapshot,
    CreditCurve,
    ModelConfig,
)


# ---------------------------------------------------------------------------
# Helpers that write temp files
# ---------------------------------------------------------------------------

def _write(tmp_path, filename, content):
    p = tmp_path / filename
    p.write_text(textwrap.dedent(content), encoding="utf-8")
    return str(p)


# ---------------------------------------------------------------------------
# load_portfolio_csv
# ---------------------------------------------------------------------------

class TestLoadPortfolioCsv:

    def test_zcb_trade_parsed_correctly(self, tmp_path):
        trades = _write(tmp_path, "trades.csv", """\
            trade_id,trade_type,netting_set_id,notional,maturity_date,start_date,receive_fixed,fixed_rate,payment_frequency
            ZCB-1,ZCB,NS-1,1000000,730,,,,
        """)
        netting = _write(tmp_path, "ns.csv", """\
            netting_set_id,counterparty_id,csa_id
            NS-1,CPTY-A,CSA-1
        """)
        result = load_portfolio_csv(trades, netting)
        assert len(result) == 1
        ns = result[0]
        assert isinstance(ns, NettingSet)
        assert ns.netting_set_id == "NS-1"
        assert ns.counterparty_id == "CPTY-A"
        assert ns.csa_id == "CSA-1"
        assert len(ns.trades) == 1
        t = ns.trades[0]
        assert isinstance(t, ZCBTa)
        assert t.trade_id == "ZCB-1"
        assert t.notional == 1_000_000.0
        assert t.maturity_date == 730

    def test_irs_trade_parsed_correctly(self, tmp_path):
        trades = _write(tmp_path, "trades.csv", """\
            trade_id,trade_type,netting_set_id,notional,maturity_date,start_date,receive_fixed,fixed_rate,payment_frequency
            IRS-1,IRS,NS-1,500000,1825,0,False,0.025,6
        """)
        netting = _write(tmp_path, "ns.csv", """\
            netting_set_id,counterparty_id,csa_id
            NS-1,CPTY-B,
        """)
        result = load_portfolio_csv(trades, netting)
        ns = result[0]
        t = ns.trades[0]
        assert isinstance(t, IRSTrade)
        assert t.trade_id == "IRS-1"
        assert t.notional == 500_000.0
        assert t.start_date == 0
        assert t.maturity_date == 1825
        assert t.receive_fixed is False
        assert t.fixed_rate == pytest.approx(0.025)
        assert t.payment_frequency == 6

    def test_irs_receive_fixed_true(self, tmp_path):
        trades = _write(tmp_path, "trades.csv", """\
            trade_id,trade_type,netting_set_id,notional,maturity_date,start_date,receive_fixed,fixed_rate,payment_frequency
            IRS-2,IRS,NS-1,200000,730,0,True,0.030,3
        """)
        netting = _write(tmp_path, "ns.csv", """\
            netting_set_id,counterparty_id,csa_id
            NS-1,CPTY-C,
        """)
        result = load_portfolio_csv(trades, netting)
        t = result[0].trades[0]
        assert t.receive_fixed is True

    def test_unknown_trade_type_skipped(self, tmp_path):
        """Rows with an unrecognised trade_type are silently skipped (continue branch)."""
        trades = _write(tmp_path, "trades.csv", """\
            trade_id,trade_type,netting_set_id,notional,maturity_date,start_date,receive_fixed,fixed_rate,payment_frequency
            FWD-1,FWD,NS-1,100000,365,,,,
            ZCB-1,ZCB,NS-1,1000000,730,,,,
        """)
        netting = _write(tmp_path, "ns.csv", """\
            netting_set_id,counterparty_id,csa_id
            NS-1,CPTY-A,
        """)
        result = load_portfolio_csv(trades, netting)
        # FWD-1 is skipped; only ZCB-1 remains
        assert len(result[0].trades) == 1
        assert result[0].trades[0].trade_id == "ZCB-1"

    def test_missing_ns_in_netting_file_falls_back_to_unknown(self, tmp_path):
        """Netting set only in trades CSV → counterparty_id defaults to UNKNOWN."""
        trades = _write(tmp_path, "trades.csv", """\
            trade_id,trade_type,netting_set_id,notional,maturity_date,start_date,receive_fixed,fixed_rate,payment_frequency
            ZCB-1,ZCB,NS-X,250000,365,,,,
        """)
        netting = _write(tmp_path, "ns.csv", """\
            netting_set_id,counterparty_id,csa_id
        """)
        result = load_portfolio_csv(trades, netting)
        assert result[0].counterparty_id == "UNKNOWN"

    def test_empty_csa_id_becomes_none(self, tmp_path):
        """An empty csa_id string resolves to None (falsy branch)."""
        trades = _write(tmp_path, "trades.csv", """\
            trade_id,trade_type,netting_set_id,notional,maturity_date,start_date,receive_fixed,fixed_rate,payment_frequency
            ZCB-1,ZCB,NS-1,1000000,365,,,,
        """)
        netting = _write(tmp_path, "ns.csv", """\
            netting_set_id,counterparty_id,csa_id
            NS-1,CPTY-A,
        """)
        result = load_portfolio_csv(trades, netting)
        assert result[0].csa_id is None

    def test_multiple_netting_sets(self, tmp_path):
        """Trades from two different netting sets produce two NettingSet objects."""
        trades = _write(tmp_path, "trades.csv", """\
            trade_id,trade_type,netting_set_id,notional,maturity_date,start_date,receive_fixed,fixed_rate,payment_frequency
            ZCB-1,ZCB,NS-1,1000000,365,,,,
            ZCB-2,ZCB,NS-2,2000000,730,,,,
        """)
        netting = _write(tmp_path, "ns.csv", """\
            netting_set_id,counterparty_id,csa_id
            NS-1,CPTY-A,CSA-1
            NS-2,CPTY-B,
        """)
        result = load_portfolio_csv(trades, netting)
        assert len(result) == 2
        ns_ids = {ns.netting_set_id for ns in result}
        assert ns_ids == {"NS-1", "NS-2"}

    def test_mixed_zcb_and_irs_in_one_ns(self, tmp_path):
        """ZCB + IRS in the same netting set are both parsed correctly."""
        trades = _write(tmp_path, "trades.csv", """\
            trade_id,trade_type,netting_set_id,notional,maturity_date,start_date,receive_fixed,fixed_rate,payment_frequency
            ZCB-1,ZCB,NS-1,1000000,730,,,,
            IRS-1,IRS,NS-1,1000000,1825,0,False,0.025,6
        """)
        netting = _write(tmp_path, "ns.csv", """\
            netting_set_id,counterparty_id,csa_id
            NS-1,CPTY-A,CSA-1
        """)
        result = load_portfolio_csv(trades, netting)
        assert len(result[0].trades) == 2
        types = {type(t) for t in result[0].trades}
        assert types == {ZCBTa, IRSTrade}


# ---------------------------------------------------------------------------
# load_discount_curve_json
# ---------------------------------------------------------------------------

class TestLoadDiscountCurveJson:

    def test_returns_curve_snapshot(self, tmp_path):
        data = {
            "currency": "USD",
            "points": [
                {"tenor": 0,   "discount_factor": 1.000},
                {"tenor": 182, "discount_factor": 0.990},
                {"tenor": 365, "discount_factor": 0.978},
            ],
        }
        path = tmp_path / "curve.json"
        path.write_text(json.dumps(data), encoding="utf-8")

        result = load_discount_curve_json(str(path))

        assert isinstance(result, CurveSnapshot)
        assert result.currency == "USD"
        assert len(result.points) == 3
        assert result.points[0].tenor == 0
        assert result.points[0].discount_factor == pytest.approx(1.0)
        assert result.points[2].tenor == 365
        assert result.points[2].discount_factor == pytest.approx(0.978)

    def test_non_usd_currency(self, tmp_path):
        data = {"currency": "EUR", "points": [{"tenor": 0, "discount_factor": 1.0}]}
        path = tmp_path / "eur.json"
        path.write_text(json.dumps(data), encoding="utf-8")
        result = load_discount_curve_json(str(path))
        assert result.currency == "EUR"

    def test_loads_real_sample_file(self):
        """Round-trip against the checked-in sample file."""
        result = load_discount_curve_json("inputs/market/discount_curve.json")
        assert isinstance(result, CurveSnapshot)
        assert result.points[0].discount_factor == pytest.approx(1.0, abs=1e-6)


# ---------------------------------------------------------------------------
# load_credit_curves_json
# ---------------------------------------------------------------------------

class TestLoadCreditCurvesJson:

    def test_returns_dict_keyed_by_entity(self, tmp_path):
        data = [
            {
                "entity_id": "CPTY-1",
                "recovery_rate": 0.40,
                "points": [
                    {"tenor": 0,   "survival_prob": 1.00},
                    {"tenor": 365, "survival_prob": 0.98},
                ],
            },
            {
                "entity_id": "BANK",
                "recovery_rate": 0.40,
                "points": [
                    {"tenor": 0,   "survival_prob": 1.00},
                    {"tenor": 365, "survival_prob": 0.995},
                ],
            },
        ]
        path = tmp_path / "credit.json"
        path.write_text(json.dumps(data), encoding="utf-8")

        result = load_credit_curves_json(str(path))

        assert set(result.keys()) == {"CPTY-1", "BANK"}
        cc = result["CPTY-1"]
        assert isinstance(cc, CreditCurve)
        assert cc.recovery_rate == pytest.approx(0.40)
        assert len(cc.points) == 2
        assert cc.points[1].survival_prob == pytest.approx(0.98)

    def test_loads_real_sample_file(self):
        result = load_credit_curves_json("inputs/market/credit_curves.json")
        assert "BANK" in result
        assert isinstance(result["BANK"], CreditCurve)


# ---------------------------------------------------------------------------
# load_model_config_json
# ---------------------------------------------------------------------------

class TestLoadModelConfigJson:

    def _make_config(self, tmp_path, extra=None):
        data = {
            "hw_params": {"mean_reversion": 0.05, "volatility": 0.01},
            "mc_config":  {"num_paths": 1000, "seed": 42, "batch_size": 200},
            "funding_spread": 0.008,
            "cost_of_capital": 0.12,
            "collateral_mode": "perfect_vm",
        }
        if extra:
            data.update(extra)
        path = tmp_path / "cfg.json"
        path.write_text(json.dumps(data), encoding="utf-8")
        return str(path)

    def test_returns_four_tuple(self, tmp_path):
        path = self._make_config(tmp_path)
        result = load_model_config_json(path)
        assert len(result) == 4

    def test_model_config_fields(self, tmp_path):
        path = self._make_config(tmp_path)
        cfg, s_f, c_cap, csa_mode = load_model_config_json(path)
        assert isinstance(cfg, ModelConfig)
        assert cfg.hw_params.mean_reversion == pytest.approx(0.05)
        assert cfg.hw_params.volatility == pytest.approx(0.01)
        assert cfg.mc_config.num_paths == 1000
        assert cfg.mc_config.seed == 42

    def test_funding_spread_and_cost_of_capital(self, tmp_path):
        path = self._make_config(tmp_path)
        _, s_f, c_cap, _ = load_model_config_json(path)
        assert s_f == pytest.approx(0.008)
        assert c_cap == pytest.approx(0.12)

    def test_collateral_mode(self, tmp_path):
        path = self._make_config(tmp_path)
        _, _, _, csa_mode = load_model_config_json(path)
        assert csa_mode == "perfect_vm"

    def test_base_currency_explicit(self, tmp_path):
        """When base_currency is present it is passed to ModelConfig."""
        path = self._make_config(tmp_path, extra={"base_currency": "EUR"})
        cfg, _, _, _ = load_model_config_json(path)
        assert cfg.base_currency == "EUR"

    def test_base_currency_defaults_to_usd(self, tmp_path):
        """When base_currency is absent ModelConfig receives the default 'USD'."""
        path = self._make_config(tmp_path)
        cfg, _, _, _ = load_model_config_json(path)
        assert cfg.base_currency == "USD"

    def test_optional_fields_default_when_absent(self, tmp_path):
        """funding_spread, cost_of_capital, collateral_mode are optional with defaults."""
        data = {
            "hw_params": {"mean_reversion": 0.03, "volatility": 0.005},
            "mc_config":  {"num_paths": 500, "seed": 7, "batch_size": 100},
        }
        path = tmp_path / "minimal.json"
        path.write_text(json.dumps(data), encoding="utf-8")
        cfg, s_f, c_cap, csa_mode = load_model_config_json(str(path))
        assert s_f == pytest.approx(0.01)
        assert c_cap == pytest.approx(0.08)
        assert csa_mode == "none"

    def test_loads_real_sample_file(self):
        cfg, s_f, c_cap, csa_mode = load_model_config_json("inputs/config/model_config.json")
        assert isinstance(cfg, ModelConfig)
        assert s_f > 0
        assert c_cap > 0
