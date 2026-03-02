from typing import List, Literal, Optional, Union
from pydantic import BaseModel, Field

# -----------------
# TRADES
# -----------------

class ZCBTa(BaseModel):
    trade_type: Literal["ZCB"] = "ZCB"
    trade_id: str
    netting_set_id: str
    notional: float
    maturity_date: int # representing days from t=0 or standard date config
    
class IRSTrade(BaseModel):
    trade_type: Literal["IRS"] = "IRS"
    trade_id: str
    netting_set_id: str
    notional: float
    start_date: int
    maturity_date: int
    receive_fixed: bool
    fixed_rate: float
    payment_frequency: int # in months (e.g. 6)
    
Trade = Union[ZCBTa, IRSTrade]

# -----------------
# NETTING SETS & CSA
# -----------------

class CSA(BaseModel):
    csa_id: str
    mode: Literal["none", "perfect_vm", "threshold"] = "none"
    threshold: float = 0.0
    mta: float = 0.0
    mpor_days: int = 0
    collateral_currency: str = "USD"

class NettingSet(BaseModel):
    netting_set_id: str
    counterparty_id: str
    csa_id: Optional[str] = None
    trades: List[Trade] = Field(..., min_length=1)

# -----------------
# MARKET DATA
# -----------------

class CurvePoint(BaseModel):
    tenor: int  # Days from t=0
    discount_factor: float

class CurveSnapshot(BaseModel):
    currency: str
    points: List[CurvePoint]

class CreditPoint(BaseModel):
    tenor: int  # Days from t=0
    survival_prob: float

class CreditCurve(BaseModel):
    entity_id: str
    recovery_rate: float
    points: List[CreditPoint]

# -----------------
# MODEL CONFIG
# -----------------

class HullWhiteParams(BaseModel):
    mean_reversion: float = Field(..., gt=0.0)
    volatility: float = Field(..., ge=0.0)

class MonteCarloConfig(BaseModel):
    num_paths: int = Field(..., gt=0)
    seed: int = 42
    batch_size: int = 1000
    dense_grid_frequency: Optional[int] = None # e.g. every 30 days

class ModelConfig(BaseModel):
    hw_params: HullWhiteParams
    mc_config: MonteCarloConfig
    base_currency: str = "USD"
