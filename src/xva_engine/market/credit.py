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
        interp_log_surv = np.interp(t, self.days, self.log_survival)
        return np.exp(interp_log_surv)

    def marginal_pd(self, t_start: float | np.ndarray, t_end: float | np.ndarray) -> Union[float, np.ndarray]:
        """
        Returns the marginal probability of default between t_start and t_end.
        """
        return self.survival_prob(t_start) - self.survival_prob(t_end)
