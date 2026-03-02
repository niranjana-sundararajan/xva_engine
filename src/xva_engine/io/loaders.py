"""
io/loaders.py
=============
CSV and JSON loaders that hydrate Pydantic schema objects from flat files.

Usage:
    netting_sets = load_portfolio_csv("inputs/trades/portfolio.csv",
                                       "inputs/netting/netting_sets.csv")
    curve        = load_discount_curve_json("inputs/market/discount_curve.json")
    credit_map   = load_credit_curves_json("inputs/market/credit_curves.json")
    config, s_f, c_cap, csa_mode = load_model_config_json("inputs/config/model_config.json")
"""
import csv
import json
from typing import Dict, List, Optional, Tuple

from .schemas import (
    ZCBTa, IRSTrade, NettingSet,
    CurveSnapshot, CurvePoint,
    CreditCurve, CreditPoint,
    Trade, ModelConfig, HullWhiteParams, MonteCarloConfig,
)


def load_portfolio_csv(trades_path: str, netting_path: str) -> List[NettingSet]:
    """
    Read portfolio.csv + netting_sets.csv and return a list of NettingSet objects.

    portfolio.csv columns (ZCB rows leave IRS-specific fields blank):
        trade_id, trade_type, netting_set_id, notional, maturity_date,
        start_date, receive_fixed, fixed_rate, payment_frequency

    netting_sets.csv columns:
        netting_set_id, counterparty_id, csa_id
    """
    # --- 1. Parse trade rows ---
    ns_trades: Dict[str, List[Trade]] = {}
    with open(trades_path, newline="") as f:
        reader = csv.DictReader(f)
        for raw in reader:
            row = {k: v.strip() for k, v in raw.items()}
            ns_id = row["netting_set_id"]
            t_type = row["trade_type"]

            if t_type == "ZCB":
                trade: Trade = ZCBTa(
                    trade_id=row["trade_id"],
                    netting_set_id=ns_id,
                    notional=float(row["notional"]),
                    maturity_date=int(row["maturity_date"]),
                )
            elif t_type == "IRS":
                trade = IRSTrade(
                    trade_id=row["trade_id"],
                    netting_set_id=ns_id,
                    notional=float(row["notional"]),
                    start_date=int(row["start_date"]),
                    maturity_date=int(row["maturity_date"]),
                    receive_fixed=row["receive_fixed"].lower() == "true",
                    fixed_rate=float(row["fixed_rate"]),
                    payment_frequency=int(row["payment_frequency"]),
                )
            else:
                continue  # unknown trade type — skip silently

            ns_trades.setdefault(ns_id, []).append(trade)

    # --- 2. Parse netting-set metadata ---
    ns_meta: Dict[str, Tuple[str, Optional[str]]] = {}
    with open(netting_path, newline="") as f:
        reader = csv.DictReader(f)
        for raw in reader:
            row = {k: v.strip() for k, v in raw.items()}
            csa_id: Optional[str] = row.get("csa_id") or None
            ns_meta[row["netting_set_id"]] = (row["counterparty_id"], csa_id)

    # --- 3. Assemble NettingSet objects ---
    result: List[NettingSet] = []
    for ns_id, trades in ns_trades.items():
        cpty_id, csa_id = ns_meta.get(ns_id, ("UNKNOWN", None))
        result.append(NettingSet(
            netting_set_id=ns_id,
            counterparty_id=cpty_id,
            csa_id=csa_id,
            trades=trades,
        ))
    return result


def load_discount_curve_json(path: str) -> CurveSnapshot:
    """Read discount_curve.json → CurveSnapshot Pydantic model."""
    with open(path) as f:
        data = json.load(f)
    return CurveSnapshot(
        currency=data["currency"],
        points=[CurvePoint(**p) for p in data["points"]],
    )


def load_credit_curves_json(path: str) -> Dict[str, CreditCurve]:
    """
    Read credit_curves.json (a JSON array of curve objects) and return a
    dict keyed by entity_id.  Must include an entry with entity_id "BANK"
    for the bank's own default curve.
    """
    with open(path) as f:
        data = json.load(f)
    return {
        item["entity_id"]: CreditCurve(
            entity_id=item["entity_id"],
            recovery_rate=item["recovery_rate"],
            points=[CreditPoint(**p) for p in item["points"]],
        )
        for item in data
    }


def load_model_config_json(path: str) -> Tuple[ModelConfig, float, float, str]:
    """
    Read model_config.json and return a 4-tuple:
        (ModelConfig, funding_spread, cost_of_capital, collateral_mode)
    """
    with open(path) as f:
        data = json.load(f)
    config = ModelConfig(
        hw_params=HullWhiteParams(**data["hw_params"]),
        mc_config=MonteCarloConfig(**data["mc_config"]),
        base_currency=data.get("base_currency", "USD"),
    )
    return (
        config,
        float(data.get("funding_spread", 0.01)),
        float(data.get("cost_of_capital", 0.08)),
        data.get("collateral_mode", "none"),
    )
