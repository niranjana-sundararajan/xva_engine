import numpy as np
import polars as pl
from ..market.curve import DiscountCurve

def compute_fva(
    grid_days: np.ndarray,
    EE: np.ndarray,
    C: np.ndarray,
    discount_curve: DiscountCurve,
    funding_spread: float  # s_f constant or array of length grid_days
) -> pl.DataFrame:
    """
    Computes FVA: funding cost on uncollateralised exposure.
    FVA = sum [ P(0, t_i) * s_f(t_i) * E[F(t_i)] * dt_i ]
    where F(t) = max(V_ns(t) - C(t), 0) => approximated by EE_net
    """
    fva_components = []

    if np.isscalar(funding_spread):
        spreads = np.full(len(grid_days), funding_spread)
    else:
        spreads = np.asarray(funding_spread)

    for i in range(1, len(grid_days)):
        t_curr = grid_days[i]
        t_prev = grid_days[i-1]
        dt_years = (t_curr - t_prev) / 365.0

        df_t = discount_curve.df(t_curr)
        ee_t = EE[i]
        sf_t = spreads[i]

        fva_incr = df_t * sf_t * ee_t * dt_years

        fva_components.append({
            "t_end_days": t_curr,
            "df": df_t,
            "EE_net": ee_t,
            "funding_spread": sf_t,
            "dt_years": dt_years,
            "fva_contribution": fva_incr
        })

    df = pl.DataFrame(fva_components)
    return df
