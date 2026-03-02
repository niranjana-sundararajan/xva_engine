import numpy as np
from ..io.schemas import ZCBTa
from ..market.curve import DiscountCurve
from ..models.hw1f import hw1f_bond_price

class ZcbPricer:
    def __init__(self, trade: ZCBTa):
        self.trade = trade

    def pv_deterministic(self, t_eval_days: float, curve: DiscountCurve) -> float:
        """
        Deterministic PV at time t. If t > maturity, PV = 0.
        Otherwise PV = Notional * P(t, T)  (where P(t, T) = df(T)/df(t))
        """
        if t_eval_days >= self.trade.maturity_date:
            return 0.0
            
        dfs_T = curve.df(self.trade.maturity_date)
        dfs_t = curve.df(t_eval_days)
        return self.trade.notional * (dfs_T / dfs_t)

    def pv_pathwise(
        self, 
        t_eval_days: float, 
        x_t: np.ndarray, 
        curve: DiscountCurve, 
        hw_a: float, 
        hw_sigma: float
    ) -> np.ndarray:
        """
        Calculates the value of the ZCB at time t for all paths.
        """
        if t_eval_days >= self.trade.maturity_date:
            return np.zeros_like(x_t)
            
        P_0_t = curve.df(t_eval_days)
        P_0_T = curve.df(self.trade.maturity_date)
        
        t_years = t_eval_days / 365.0
        T_years = self.trade.maturity_date / 365.0
        
        bond_prices = hw1f_bond_price(t_years, T_years, x_t, P_0_t, P_0_T, hw_a, hw_sigma)
        return self.trade.notional * bond_prices
