import numpy as np
from typing import Union
from ..io.schemas import CreditCurve

class CreditCurveModel:
    """Provides log-linear interpolation for survival probabilities."""

    def __init__(self, credit_data: CreditCurve):
        self.entity_id = credit_data.entity_id
        self.recovery_rate = credit_data.recovery_rate

        points_sorted = sorted(credit_data.points, key=lambda p: p.tenor)
        self.days = np.array([p.tenor for p in points_sorted], dtype=np.float64)
        self.survival = np.array([p.survival_prob for p in points_sorted], dtype=np.float64)

        # log survival for interpolation
        self.log_survival = np.log(np.maximum(self.survival, 1e-12))

    def survival_prob(self, t: float | np.ndarray) -> Union[float, np.ndarray]:
        """
        Evaluate survival probability at time t (in days).
        Extrapolates flat hazard rate natively via linear log-survival extension.
        """
        t_arr = np.asarray(t, dtype=np.float64)

        if self.days.size == 1:
            log_surv = np.full_like(t_arr, self.log_survival[0], dtype=np.float64)
            result = np.exp(log_surv)
            return result if isinstance(t, np.ndarray) or (isinstance(t_arr, np.ndarray) and t_arr.ndim > 0) else float(result)

        first_slope = (self.log_survival[1] - self.log_survival[0]) / (self.days[1] - self.days[0])
        last_slope = (self.log_survival[-1] - self.log_survival[-2]) / (self.days[-1] - self.days[-2])

        log_surv = np.empty_like(t_arr, dtype=np.float64)

        # Interpolation inside bounds
        inside_mask = (t_arr >= self.days[0]) & (t_arr <= self.days[-1])
        if np.any(inside_mask):
            log_surv[inside_mask] = np.interp(
                t_arr[inside_mask],
                self.days,
                self.log_survival,
            )

        # Extrapolation outside bounds
        left_mask = t_arr < self.days[0]
        if np.any(left_mask):
            log_surv[left_mask] = self.log_survival[0] + first_slope * (t_arr[left_mask] - self.days[0])

        right_mask = t_arr > self.days[-1]
        if np.any(right_mask):
            log_surv[right_mask] = self.log_survival[-1] + last_slope * (t_arr[right_mask] - self.days[-1])

        result = np.exp(log_surv)
        return result if isinstance(t, np.ndarray) or (isinstance(t_arr, np.ndarray) and t_arr.ndim > 0) else float(result)

    def marginal_pd(self, t_start: float | np.ndarray, t_end: float | np.ndarray) -> Union[float, np.ndarray]:
        """
        Returns the marginal probability of default between t_start and t_end.
        """
        return self.survival_prob(t_start) - self.survival_prob(t_end)
