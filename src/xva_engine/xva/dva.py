import numpy as np
import polars as pl
from ..market.curve import DiscountCurve
from ..market.credit import CreditCurveModel

def compute_dva(
    grid_days: np.ndarray,
    ENE: np.ndarray,
    discount_curve: DiscountCurve,
    bank_credit: CreditCurveModel
) -> pl.DataFrame:
    """
    Computes DVA and returns a Polars DataFrame with contribution tables.
    DVA = (1 - R_b) * sum [ P(0, t_i) * ENE(t_i) * dPD_b(t_i) ]
    """
    dva_components = []
    
    LGD_b = 1.0 - bank_credit.recovery_rate
    
    for i in range(1, len(grid_days)):
        t_prev = grid_days[i-1]
        t_curr = grid_days[i]
        
        dPD_b = bank_credit.marginal_pd(t_prev, t_curr)
        df_t = discount_curve.df(t_curr)
        ene_t = abs(ENE[i])  # ENE <= 0; DVA uses its magnitude
        
        dva_incr = LGD_b * df_t * ene_t * dPD_b
        
        dva_components.append({
            "t_end_days": t_curr,
            "df": df_t,
            "ENE": ene_t,
            "dPD_b": dPD_b,
            "dva_contribution": dva_incr
        })
        
    df = pl.DataFrame(dva_components)
    return df
