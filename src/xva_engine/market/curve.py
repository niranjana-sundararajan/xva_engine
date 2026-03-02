import numpy as np
from typing import List, Tuple, Union
from ..io.schemas import CurveSnapshot

class DiscountCurve:
    """Provides log-linear interpolation for discount factors."""

    def __init__(self, curve_data: CurveSnapshot):
        self.currency = curve_data.currency
        points = curve_data.points
        
        has_zero = any(p.tenor == 0 for p in points)
        if not has_zero:
            raise ValueError("Discount curve must contain a pillar for tenor=0.")

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
        t_arr = np.asarray(t, dtype=np.float64)

        if self.days.size == 1:
            log_df = np.full_like(t_arr, self.log_dfs[0], dtype=np.float64)
            result = np.exp(log_df)
            return result if isinstance(t, np.ndarray) or (isinstance(t_arr, np.ndarray) and t_arr.ndim > 0) else float(result)

        first_slope = (self.log_dfs[1] - self.log_dfs[0]) / (self.days[1] - self.days[0])
        last_slope = (self.log_dfs[-1] - self.log_dfs[-2]) / (self.days[-1] - self.days[-2])

        log_df = np.empty_like(t_arr, dtype=np.float64)

        # Interpolation inside bounds
        inside_mask = (t_arr >= self.days[0]) & (t_arr <= self.days[-1])
        if np.any(inside_mask):
            log_df[inside_mask] = np.interp(
                t_arr[inside_mask],
                self.days,
                self.log_dfs,
            )

        # Extrapolation outside bounds
        left_mask = t_arr < self.days[0]
        if np.any(left_mask):
            log_df[left_mask] = self.log_dfs[0] + first_slope * (t_arr[left_mask] - self.days[0])

        right_mask = t_arr > self.days[-1]
        if np.any(right_mask):
            log_df[right_mask] = self.log_dfs[-1] + last_slope * (t_arr[right_mask] - self.days[-1])

        result = np.exp(log_df)
        return result if isinstance(t, np.ndarray) or (isinstance(t_arr, np.ndarray) and t_arr.ndim > 0) else float(result)
        t_years = np.maximum(t / 365.0, 1e-6) # avoid div by zero
        return -np.log(self.df(t)) / t_years
