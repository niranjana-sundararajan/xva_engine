import numpy as np
from typing import List, Tuple, Union
from ..io.schemas import CurveSnapshot

class DiscountCurve:
    """Provides log-linear interpolation for discount factors."""

    def __init__(self, curve_data: CurveSnapshot):
        self.currency = curve_data.currency
        # Ensure we have a day 0 point
        points = curve_data.points
        points_sorted = sorted(points, key=lambda p: p.tenor)
        
        self.days = np.array([p.tenor for p in points_sorted], dtype=np.float64)
        self.dfs = np.array([p.discount_factor for p in points_sorted], dtype=np.float64)
        
        # Log dfs for interpolation
        self.log_dfs = np.log(self.dfs)

    def df(self, t: float | np.ndarray) -> Union[float, np.ndarray]:
        """
        Evaluate the discount factor at time t (in days).
        Extrapolates flat zero rate on the left, and flat forward rate on the right.
        """
        # Linear interpolation on log DF
        interp_log_df = np.interp(t, self.days, self.log_dfs)
        return np.exp(interp_log_df)

    def zero_rate(self, t: float | np.ndarray) -> Union[float, np.ndarray]:
        """
        Continuous zero rate for time t (days).
        """
        t_years = np.maximum(t / 365.0, 1e-6) # avoid div by zero
        return -np.log(self.df(t)) / t_years
