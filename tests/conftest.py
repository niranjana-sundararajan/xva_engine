"""
Root conftest.py – sets NUMBA_DISABLE_JIT=1 before any test module is imported
so that numba @njit functions behave as plain Python and are measurable by
pytest-cov branch coverage.
"""
import os
os.environ.setdefault("NUMBA_DISABLE_JIT", "1")

import numpy as np
import pytest
from xva_engine.io.schemas import (
    ZCBTa, IRSTrade, NettingSet, CSA,
    CurveSnapshot, CurvePoint, CreditCurve, CreditPoint,
    HullWhiteParams, MonteCarloConfig, ModelConfig,
)
from xva_engine.market.curve import DiscountCurve
from xva_engine.market.credit import CreditCurveModel


# ---------------------------------------------------------------------------
# Shared market fixtures
# ---------------------------------------------------------------------------

@pytest.fixture(scope="session")
def sample_curve_snapshot():
    return CurveSnapshot(
        currency="USD",
        points=[
            CurvePoint(tenor=0, discount_factor=1.0),
            CurvePoint(tenor=182, discount_factor=0.988),
            CurvePoint(tenor=365, discount_factor=0.975),
            CurvePoint(tenor=730, discount_factor=0.950),
            CurvePoint(tenor=1825, discount_factor=0.880),
        ],
    )


@pytest.fixture(scope="session")
def flat_curve(sample_curve_snapshot):
    return DiscountCurve(sample_curve_snapshot)


@pytest.fixture(scope="session")
def sample_cpty_credit():
    return CreditCurve(
        entity_id="CPTY-1",
        recovery_rate=0.40,
        points=[
            CreditPoint(tenor=0, survival_prob=1.0),
            CreditPoint(tenor=365, survival_prob=0.98),
            CreditPoint(tenor=730, survival_prob=0.96),
            CreditPoint(tenor=1825, survival_prob=0.90),
        ],
    )


@pytest.fixture(scope="session")
def sample_bank_credit():
    return CreditCurve(
        entity_id="BANK",
        recovery_rate=0.40,
        points=[
            CreditPoint(tenor=0, survival_prob=1.0),
            CreditPoint(tenor=365, survival_prob=0.99),
            CreditPoint(tenor=730, survival_prob=0.985),
            CreditPoint(tenor=1825, survival_prob=0.96),
        ],
    )


@pytest.fixture(scope="session")
def cpty_credit_model(sample_cpty_credit):
    return CreditCurveModel(sample_cpty_credit)


@pytest.fixture(scope="session")
def bank_credit_model(sample_bank_credit):
    return CreditCurveModel(sample_bank_credit)


# ---------------------------------------------------------------------------
# Shared trade / netting-set fixtures
# ---------------------------------------------------------------------------

@pytest.fixture(scope="session")
def zcb_trade():
    return ZCBTa(
        trade_id="ZCB-1",
        netting_set_id="NS-1",
        notional=1_000_000,
        maturity_date=730,
    )


@pytest.fixture(scope="session")
def irs_payer_trade():
    """Payer-fixed IRS: receive_fixed=False."""
    return IRSTrade(
        trade_id="IRS-1",
        netting_set_id="NS-1",
        notional=1_000_000,
        start_date=0,
        maturity_date=1825,
        receive_fixed=False,
        fixed_rate=0.025,
        payment_frequency=6,
    )


@pytest.fixture(scope="session")
def irs_receiver_trade():
    """Receiver-fixed IRS: receive_fixed=True."""
    return IRSTrade(
        trade_id="IRS-2",
        netting_set_id="NS-1",
        notional=1_000_000,
        start_date=0,
        maturity_date=1825,
        receive_fixed=True,
        fixed_rate=0.025,
        payment_frequency=6,
    )


@pytest.fixture(scope="session")
def mixed_netting_set(zcb_trade, irs_payer_trade):
    return NettingSet(
        netting_set_id="NS-1",
        counterparty_id="CPTY-1",
        trades=[zcb_trade, irs_payer_trade],
    )


# ---------------------------------------------------------------------------
# Shared HW params / MC config
# ---------------------------------------------------------------------------

@pytest.fixture(scope="session")
def hw_params():
    return HullWhiteParams(mean_reversion=0.05, volatility=0.01)


@pytest.fixture(scope="session")
def mc_config():
    return MonteCarloConfig(num_paths=2000, seed=42, batch_size=500)


@pytest.fixture(scope="session")
def model_config(hw_params, mc_config):
    return ModelConfig(hw_params=hw_params, mc_config=mc_config)
