"""
XVA Engine: Main orchestrator that wires all components together.
"""

import uuid
import numpy as np
from typing import Optional

from .io.schemas import (
    NettingSet,
    CurveSnapshot,
    CreditCurve,
    ModelConfig,
    ZCBTa,
    IRSTrade,
    CSA,
)
from .market.curve import DiscountCurve
from .market.credit import CreditCurveModel
from .products.zcb import ZcbPricer
from .products.irs import IrsPricer
from .sim.timegrid import build_simulation_grid
from .sim.batching import MonteCarloEngine
from .exposure.collateral import calculate_collateral
from .exposure.exposure import calculate_exposures
from .xva.cva import compute_cva
from .xva.dva import compute_dva
from .xva.fva import compute_fva
from .xva.mva import compute_mva, PercentileIM
from .xva.kva import compute_kva, EECapital
from .explain.bundle import ExplainabilityBundle


class XVAEngine:
    """
    Unified XVA Engine: runs Monte Carlo simulation and computes all XVA metrics.
    """

    def __init__(
        self,
        netting_set: NettingSet,
        curve_snapshot: CurveSnapshot,
        cpty_credit: CreditCurve,
        bank_credit: CreditCurve,
        config: ModelConfig,
        funding_spread: float = 0.01,
        cost_of_capital: float = 0.08,
        output_dir: str = "output",
    ):
        self.netting_set = netting_set
        self.config = config
        self.funding_spread = funding_spread
        self.cost_of_capital = cost_of_capital
        self.output_dir = output_dir

        # Market objects
        self.discount_curve = DiscountCurve(curve_snapshot)
        self.cpty_credit_model = CreditCurveModel(cpty_credit)
        self.bank_credit_model = CreditCurveModel(bank_credit)

        # Build product pricers
        self.pricers = []
        for trade in netting_set.trades:
            if trade.trade_type == "ZCB":
                self.pricers.append(ZcbPricer(trade))
            elif trade.trade_type == "IRS":
                self.pricers.append(IrsPricer(trade))
            else:  # pragma: no cover
                pass

        # CSA lookup
        if netting_set.trades:
            _extra = getattr(netting_set.trades[0], "model_extra", None)
            self.csa = _extra.get("csa", None) if _extra else None
        else:  # pragma: no cover
            self.csa = None  # pragma: no cover

        if self.csa:  # pragma: no cover
            self.csa_schema = CSA(**self.csa)  # pragma: no cover
        elif netting_set.csa_id:
            # Provide a fallback strictly if a csa_id is given but no detail exists
            self.csa_schema = CSA(csa_id=netting_set.csa_id, mode="perfect_vm")
        else:
            self.csa_schema = None

        self.run_id = str(uuid.uuid4())[:8]

    def run(self) -> dict:
        """
        Run the full XVA engine. Returns dict of XVA scalars and contribution tables.
        """
        mc_cfg = self.config.mc_config
        hw = self.config.hw_params

        # Build time grid
        grid_days = build_simulation_grid(
            self.netting_set, dense_frequency_days=mc_cfg.dense_grid_frequency
        )

        # Simulate HW paths
        mc_engine = MonteCarloEngine(
            num_paths=mc_cfg.num_paths,
            seed=mc_cfg.seed,
            hw_a=hw.mean_reversion,
            hw_sigma=hw.volatility,
        )
        x_paths = mc_engine.simulate_paths(grid_days)

        # Pathwise valuation — accumulate per trade type and as netting-set total
        V_by_type: dict = {}
        for pricer in self.pricers:
            t_type = pricer.trade.trade_type
            if t_type not in V_by_type:
                V_by_type[t_type] = np.zeros_like(x_paths)
            for i, t_days in enumerate(grid_days):
                pv = pricer.pv_pathwise(
                    t_eval_days=t_days,
                    x_t=x_paths[i, :],
                    curve=self.discount_curve,
                    hw_a=hw.mean_reversion,
                    hw_sigma=hw.volatility,
                )
                V_by_type[t_type][i, :] += pv

        V_ns_paths = np.zeros_like(x_paths)
        for _V in V_by_type.values():
            V_ns_paths = V_ns_paths + _V

        # Collateral (CSA-driven)
        C_paths = calculate_collateral(V_ns_paths, self.csa_schema)

        # Exposures — netting-set level
        exp = calculate_exposures(V_ns_paths, C_paths)
        EE = exp["EE"]
        ENE = exp["ENE"]

        # Per-product-type EE (for UI breakdown by product)
        ee_by_type = {
            t_type: calculate_exposures(V, C_paths)["EE"]
            for t_type, V in V_by_type.items()
        }

        # XVA computation
        cva_df = compute_cva(grid_days, EE, self.discount_curve, self.cpty_credit_model)
        dva_df = compute_dva(
            grid_days, ENE, self.discount_curve, self.bank_credit_model
        )
        fva_df = compute_fva(
            grid_days,
            EE,
            np.zeros(len(grid_days)),
            self.discount_curve,
            self.funding_spread,
        )
        mva_df = compute_mva(
            grid_days,
            V_ns_paths,
            self.discount_curve,
            self.funding_spread,
            im_model=PercentileIM(),
        )
        kva_df = compute_kva(
            grid_days,
            V_ns_paths,
            self.discount_curve,
            self.cost_of_capital,
            capital_model=EECapital(),
        )

        CVA = float(cva_df["cva_contribution"].sum())
        DVA = float(dva_df["dva_contribution"].sum())
        FVA = float(fva_df["fva_contribution"].sum())
        MVA = float(mva_df["mva_contribution"].sum())
        KVA = float(kva_df["kva_contribution"].sum())

        # Bundle
        bundle = ExplainabilityBundle(self.output_dir, self.run_id)
        bundle.save_config(
            {
                "run_id": self.run_id,
                "num_paths": mc_cfg.num_paths,
                "seed": mc_cfg.seed,
                "hw_a": hw.mean_reversion,
                "hw_sigma": hw.volatility,
                "funding_spread": self.funding_spread,
                "cost_of_capital": self.cost_of_capital,
            }
        )
        bundle.save_array("x_paths_sample", x_paths[:, : min(100, mc_cfg.num_paths)])
        bundle.save_array(
            "V_ns_paths_sample", V_ns_paths[:, : min(100, mc_cfg.num_paths)]
        )
        bundle.save_array("EE", EE)
        bundle.save_array("ENE", ENE)
        bundle.save_table("cva_table", cva_df)
        bundle.save_table("dva_table", dva_df)
        bundle.save_table("fva_table", fva_df)
        bundle.save_table("mva_table", mva_df)
        bundle.save_table("kva_table", kva_df)
        bundle.add_section(
            "XVA Summary",
            f"CVA: {CVA:.6f}\nDVA: {DVA:.6f}\nFVA: {FVA:.6f}\nMVA: {MVA:.6f}\nKVA: {KVA:.6f}",
        )
        bundle.write_math_report()

        return {
            "run_id": self.run_id,
            "grid_days": grid_days,
            "EE": EE,
            "ENE": ENE,
            "CVA": CVA,
            "DVA": DVA,
            "FVA": FVA,
            "MVA": MVA,
            "KVA": KVA,
            "cva_table": cva_df,
            "dva_table": dva_df,
            "fva_table": fva_df,
            "mva_table": mva_df,
            "kva_table": kva_df,
            # Sample paths for UI visualisation (<=50 paths to keep memory light)
            "V_ns_sample": V_ns_paths[:, : min(50, mc_cfg.num_paths)],
            "x_sample": x_paths[:, : min(50, mc_cfg.num_paths)],
            "ee_by_type": ee_by_type,
            "V_sample_by_type": {
                t: V[:, : min(50, mc_cfg.num_paths)] for t, V in V_by_type.items()
            },
        }
