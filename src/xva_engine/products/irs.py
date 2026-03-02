import numpy as np
from ..io.schemas import IRSTrade
from ..market.curve import DiscountCurve
from ..models.hw1f import hw1f_bond_price
from .schedule import build_irs_schedule, year_fraction_act365f

class IrsPricer:
    def __init__(self, trade: IRSTrade):
        self.trade = trade
        self.schedule = build_irs_schedule(trade.start_date, trade.maturity_date, trade.payment_frequency)
        
        # Calculate alphas (year fractions)
        self.alphas = []
        prev_date = trade.start_date
        for d in self.schedule:
            self.alphas.append(year_fraction_act365f(prev_date, d))
            prev_date = d
        self.alphas = np.array(self.alphas)

    def pv_deterministic(self, t_eval_days: float, curve: DiscountCurve) -> float:
        if t_eval_days >= self.trade.maturity_date:
            return 0.0

        # Fixed Leg
        pv_fixed = 0.0
        prev_date = self.trade.start_date
        
        # Find next payment dates
        valid_idx = self.schedule > t_eval_days
        remaining_dates = self.schedule[valid_idx]
        remaining_alphas = self.alphas[valid_idx]
        
        if len(remaining_dates) == 0:  # pragma: no cover
            return 0.0

        dfs_t = curve.df(t_eval_days)
        
        for d, alpha in zip(remaining_dates, remaining_alphas):
            df_d = curve.df(d)
            P_t_d = df_d / dfs_t
            pv_fixed += self.trade.notional * alpha * self.trade.fixed_rate * P_t_d

        # Float Leg (Par approximation)
        # PV_flt = N * (1 - P(t, T_end)) + accrual if between payment dates (ignoring stub for MVP math engine)
        T_end = self.trade.maturity_date
        df_T_end = curve.df(T_end)
        P_t_Tend = df_T_end / dfs_t
        pv_float = self.trade.notional * (1.0 - P_t_Tend)

        if self.trade.receive_fixed:
            return pv_fixed - pv_float
        else:
            return pv_float - pv_fixed

    def pv_pathwise(
        self, 
        t_eval_days: float, 
        x_t: np.ndarray, 
        curve: DiscountCurve, 
        hw_a: float, 
        hw_sigma: float
    ) -> np.ndarray:
        
        if t_eval_days >= self.trade.maturity_date:
            return np.zeros_like(x_t)

        valid_idx = self.schedule > t_eval_days
        remaining_dates = self.schedule[valid_idx]
        remaining_alphas = self.alphas[valid_idx]
        
        if len(remaining_dates) == 0:  # pragma: no cover
            return np.zeros_like(x_t)

        t_years = t_eval_days / 365.0
        P_0_t = curve.df(t_eval_days)

        # Fixed Leg
        pv_fixed = np.zeros_like(x_t)
        for d, alpha in zip(remaining_dates, remaining_alphas):
            d_years = d / 365.0
            P_0_d = curve.df(d)
            bond_prices = hw1f_bond_price(t_years, d_years, x_t, P_0_t, P_0_d, hw_a, hw_sigma)
            pv_fixed += self.trade.notional * alpha * self.trade.fixed_rate * bond_prices

        # Float Leg (Par approx MVP)
        T_end = self.trade.maturity_date
        T_end_years = T_end / 365.0
        P_0_Tend = curve.df(T_end)
        bond_prices_Tend = hw1f_bond_price(t_years, T_end_years, x_t, P_0_t, P_0_Tend, hw_a, hw_sigma)
        pv_float = self.trade.notional * (1.0 - bond_prices_Tend)

        if self.trade.receive_fixed:
            return pv_fixed - pv_float
        else:
            return pv_float - pv_fixed
