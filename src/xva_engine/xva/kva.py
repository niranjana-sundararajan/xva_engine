import numpy as np
import polars as pl
from typing import Union
from ..market.curve import DiscountCurve


class CapitalModel:
    """
    Base capital model — returns zero capital for all paths.
    Subclass to provide a concrete capital estimate.
    """
    def capital(self, t_days: float, V_ns_paths: np.ndarray) -> np.ndarray:
        return np.zeros(V_ns_paths.shape[1] if V_ns_paths.ndim == 2 else len(V_ns_paths))


class EECapital(CapitalModel):
    """
    Exposure-based capital requirement.

    K(t, path) = max(V_NS(t, path), 0) * rwa_factor

    The rwa_factor encodes the capital stack: credit risk weight times capital
    ratio (default 0.08, representing 8% capital ratio against counterparty
    credit exposure).
    """
    def __init__(self, rwa_factor: float = 0.08):
        self.rwa_factor = rwa_factor

    def capital(self, t_days: float, V_ns_paths: np.ndarray) -> np.ndarray:
        return np.maximum(V_ns_paths, 0.0) * self.rwa_factor

def compute_kva(
    grid_days: np.ndarray,
    V_ns_paths: np.ndarray,
    discount_curve: DiscountCurve,
    cost_of_capital: Union[float, np.ndarray],
    capital_model: CapitalModel = None
) -> pl.DataFrame:
    """
    Computes KVA using a Capital model interface.
    KVA = sum [ P(0, t_i) * c_cap(t_i) * E[K(t_i)] * dt_i ]
    """
    if capital_model is None:
        capital_model = CapitalModel()

    if np.isscalar(cost_of_capital):
        costs = np.full(len(grid_days), cost_of_capital)
    else:
        costs = np.asarray(cost_of_capital)
        if len(costs) != len(grid_days):
            raise ValueError(
                f"cost_of_capital array length {len(costs)} must match grid_days length {len(grid_days)}"
            )

    kva_components = []
    for i in range(1, len(grid_days)):
        t_curr = grid_days[i]
        t_prev = grid_days[i-1]
        dt_years = (t_curr - t_prev) / 365.0

        k_paths = capital_model.capital(t_curr, V_ns_paths[i, :])
        E_k = np.mean(k_paths)
        df_t = discount_curve.df(t_curr)
        c_t = costs[i]

        kva_incr = df_t * c_t * E_k * dt_years

        kva_components.append({
            "t_end_days": t_curr,
            "df": df_t,
            "E_K": E_k,
            "cost_of_capital": c_t,
            "dt_years": dt_years,
            "kva_contribution": kva_incr
        })

    df = pl.DataFrame(kva_components)
    return df
