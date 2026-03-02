import numpy as np
import polars as pl
from ..market.curve import DiscountCurve
from ..market.credit import CreditCurveModel

def compute_cva(
    grid_days: np.ndarray,
    EE: np.ndarray,
    discount_curve: DiscountCurve,
    cpty_credit: CreditCurveModel
) -> pl.DataFrame:
    """
    Computes CVA and returns a Polars DataFrame with contribution tables.
    CVA = (1 - R) * sum [ P(0, t_i) * EE(t_i) * dPD(t_i) ]
    """
    cva_components = []
    
    # We evaluate terms at the grid steps
    LGD = 1.0 - cpty_credit.recovery_rate
    
    for i in range(1, len(grid_days)):
        t_prev = grid_days[i-1]
        t_curr = grid_days[i]
        
        # Marginal PD between t_prev and t_curr
        dPD = cpty_credit.marginal_pd(t_prev, t_curr)
        
        # DF at the bucket center or end (we use end for simplicity)
        df_t = discount_curve.df(t_curr)
        
        # EE at bucket end
        ee_t = EE[i]
        
        cva_incr = LGD * df_t * ee_t * dPD
        
        cva_components.append({
            "t_end_days": t_curr,
            "df": df_t,
            "EE": ee_t,
            "dPD": dPD,
            "cva_contribution": cva_incr
        })
        
    df = pl.DataFrame(cva_components)
    return df
