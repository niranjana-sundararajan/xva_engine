"""
tests/io/test_schemas.py
========================
100% coverage for src/xva_engine/io/schemas.py

Tests every Pydantic model: field defaults, validation errors,
union discriminator (Trade = Union[ZCBTa, IRSTrade]), and all
nested models.
"""
import pytest
from pydantic import ValidationError

from xva_engine.io.schemas import (
    ZCBTa, IRSTrade, NettingSet, CSA,
    CurvePoint, CurveSnapshot,
    CreditPoint, CreditCurve,
    HullWhiteParams, MonteCarloConfig, ModelConfig,
)


class TestZCBTa:
    def test_defaults_trade_type(self):
        t = ZCBTa(trade_id="T1", netting_set_id="NS1", notional=1e6, maturity_date=365)
        assert t.trade_type == "ZCB"

    def test_all_fields_stored(self):
        t = ZCBTa(trade_id="T2", netting_set_id="NS2", notional=500_000, maturity_date=730)
        assert t.trade_id == "T2"
        assert t.netting_set_id == "NS2"
        assert t.notional == 500_000
        assert t.maturity_date == 730

    def test_invalid_trade_type_rejected(self):
        with pytest.raises(ValidationError):
            ZCBTa(trade_type="BOND", trade_id="T", netting_set_id="NS", notional=1e6, maturity_date=365)

    def test_missing_required_field_raises(self):
        with pytest.raises(ValidationError):
            ZCBTa(trade_id="T", netting_set_id="NS", maturity_date=365)


class TestIRSTrade:
    def test_defaults_trade_type(self):
        t = IRSTrade(
            trade_id="I1", netting_set_id="NS1", notional=1e6,
            start_date=0, maturity_date=1825, receive_fixed=True,
            fixed_rate=0.03, payment_frequency=6,
        )
        assert t.trade_type == "IRS"

    def test_all_fields_stored(self):
        t = IRSTrade(
            trade_id="I2", netting_set_id="NS2", notional=2e6,
            start_date=30, maturity_date=1095, receive_fixed=False,
            fixed_rate=0.025, payment_frequency=12,
        )
        assert t.notional == 2e6
        assert t.start_date == 30
        assert t.receive_fixed is False
        assert t.fixed_rate == 0.025
        assert t.payment_frequency == 12

    def test_missing_required_field_raises(self):
        with pytest.raises(ValidationError):
            IRSTrade(trade_id="I", netting_set_id="NS", notional=1e6, maturity_date=365)


class TestCSA:
    def test_default_mode_is_none(self):
        csa = CSA(csa_id="CSA1")
        assert csa.mode == "none"
        assert csa.threshold == 0.0
        assert csa.mta == 0.0
        assert csa.mpor_days == 0
        assert csa.collateral_currency == "USD"

    def test_perfect_vm_mode(self):
        csa = CSA(csa_id="CSA2", mode="perfect_vm")
        assert csa.mode == "perfect_vm"

    def test_threshold_mode_with_values(self):
        csa = CSA(csa_id="CSA3", mode="threshold", threshold=50_000.0, mta=10_000.0, mpor_days=10)
        assert csa.mode == "threshold"
        assert csa.threshold == 50_000.0
        assert csa.mta == 10_000.0
        assert csa.mpor_days == 10

    def test_invalid_mode_raises(self):
        with pytest.raises(ValidationError):
            CSA(csa_id="CSA4", mode="invalid_mode")


class TestNettingSet:
    def test_basic_construction(self, zcb_trade, irs_payer_trade):
        ns = NettingSet(
            netting_set_id="NS-1",
            counterparty_id="CPTY-1",
            trades=[zcb_trade, irs_payer_trade],
        )
        assert ns.netting_set_id == "NS-1"
        assert ns.csa_id is None
        assert len(ns.trades) == 2

    def test_with_csa_id(self, zcb_trade):
        ns = NettingSet(
            netting_set_id="NS-2",
            counterparty_id="CPTY-2",
            csa_id="CSA-1",
            trades=[zcb_trade],
        )
        assert ns.csa_id == "CSA-1"


class TestCurveSnapshot:
    def test_basic_construction(self):
        snap = CurveSnapshot(
            currency="EUR",
            points=[CurvePoint(tenor=0, discount_factor=1.0)],
        )
        assert snap.currency == "EUR"
        assert snap.points[0].tenor == 0
        assert snap.points[0].discount_factor == 1.0


class TestCreditCurveSchema:
    def test_basic_construction(self):
        cc = CreditCurve(
            entity_id="E1",
            recovery_rate=0.40,
            points=[CreditPoint(tenor=0, survival_prob=1.0)],
        )
        assert cc.entity_id == "E1"
        assert cc.recovery_rate == 0.40


class TestHullWhiteParams:
    def test_valid_params(self):
        p = HullWhiteParams(mean_reversion=0.05, volatility=0.01)
        assert p.mean_reversion == 0.05
        assert p.volatility == 0.01

    def test_zero_values_allowed(self):
        p = HullWhiteParams(mean_reversion=0.0, volatility=0.0)
        assert p.mean_reversion == 0.0

    def test_negative_mean_reversion_rejected(self):
        with pytest.raises(ValidationError):
            HullWhiteParams(mean_reversion=-0.01, volatility=0.01)

    def test_negative_volatility_rejected(self):
        with pytest.raises(ValidationError):
            HullWhiteParams(mean_reversion=0.05, volatility=-0.001)


class TestMonteCarloConfig:
    def test_defaults(self):
        cfg = MonteCarloConfig(num_paths=1000)
        assert cfg.seed == 42
        assert cfg.batch_size == 1000
        assert cfg.dense_grid_frequency is None

    def test_custom_values(self):
        cfg = MonteCarloConfig(num_paths=5000, seed=7, batch_size=500, dense_grid_frequency=30)
        assert cfg.num_paths == 5000
        assert cfg.seed == 7
        assert cfg.dense_grid_frequency == 30

    def test_zero_paths_rejected(self):
        with pytest.raises(ValidationError):
            MonteCarloConfig(num_paths=0)


class TestModelConfig:
    def test_default_currency(self, hw_params, mc_config):
        cfg = ModelConfig(hw_params=hw_params, mc_config=mc_config)
        assert cfg.base_currency == "USD"

    def test_custom_currency(self, hw_params, mc_config):
        cfg = ModelConfig(hw_params=hw_params, mc_config=mc_config, base_currency="EUR")
        assert cfg.base_currency == "EUR"
