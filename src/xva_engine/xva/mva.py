import numpy as np
import polars as pl
from scipy.special import ndtri
from ..market.curve import DiscountCurve


class InitialMarginModel:
    """
    Base IM model — returns zero initial margin for all paths.
    Subclass to provide a concrete IM estimate.
    """
    def im(self, t_days: float, V_ns_paths: np.ndarray) -> np.ndarray:
        return np.zeros(V_ns_paths.shape[1] if V_ns_paths.ndim == 2 else len(V_ns_paths))


class PercentileIM(InitialMarginModel):
    """
    Normal-distribution IM approximation.

    IM(t) = z_alpha * std(V_NS(t)) * sqrt(MPOR / 252)

    Applied uniformly across all paths. This is equivalent to a
    volatility-scaled SIMM-like margin over the Margin Period of Risk.
    """
    def __init__(self, confidence: float = 0.99, mpor_days: int = 10):
        self._z = float(ndtri(confidence))
        self._mpor_factor = (mpor_days / 252.0) ** 0.5

    def im(self, t_days: float, V_ns_paths: np.ndarray) -> np.ndarray:
        std_v = float(np.std(V_ns_paths))
        im_val = max(0.0, self._z * std_v * self._mpor_factor)
        return np.full(len(V_ns_paths), im_val)

def compute_mva(
    grid_days: np.ndarray,
    V_ns_paths: np.ndarray,
    discount_curve: DiscountCurve,
    funding_spread: float,
    im_model: InitialMarginModel = None
) -> pl.DataFrame:
    """
    Computes MVA using an IM model interface.
    MVA = sum [ P(0, t_i) * s_f(t_i) * E[IM(t_i)] * dt_i ]
    """
    if im_model is None:
        im_model = InitialMarginModel()

    if np.isscalar(funding_spread):
        spreads = np.full(len(grid_days), funding_spread)
    else:
        spreads = np.asarray(funding_spread)

    mva_components = []
    for i in range(1, len(grid_days)):
        t_curr = grid_days[i]
        t_prev = grid_days[i-1]
        dt_years = (t_curr - t_prev) / 365.0

        im_paths = im_model.im(t_curr, V_ns_paths[i, :])
        E_im = np.mean(im_paths)
        df_t = discount_curve.df(t_curr)
        sf_t = spreads[i]

        mva_incr = df_t * sf_t * E_im * dt_years

        mva_components.append({
            "t_end_days": t_curr,
            "df": df_t,
            "E_IM": E_im,
            "funding_spread": sf_t,
            "dt_years": dt_years,
            "mva_contribution": mva_incr
        })

    df = pl.DataFrame(mva_components)
    return df
