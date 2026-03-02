import numpy as np
from numba import njit
from typing import Union

@njit
def B_func(t: float, T: float, a: float) -> float:
    """
    B(t, T) in Hull-White 1F
    Returns 0 if t >= T
    """
    if t >= T:
        return 0.0
    # t and T are in years for HW formulas
    dt = T - t
    return (1.0 - np.exp(-a * dt)) / a

@njit
def V_func(t: float, T: float, a: float, sigma: float) -> float:
    """
    Variance term V(t, T) for Hull-White 1F bond reconstruction.
    """
    if t >= T:
        return 0.0
    dt = T - t
    B = (1.0 - np.exp(-a * dt)) / a
    return (sigma**2 / (2 * a)) * (B**2) * (1.0 - np.exp(-2 * a * t))

def hw1f_bond_price(
    t: float, # Evaluation time in years
    T_maturity: float, # Maturity time in years
    x_t: np.ndarray, # State variable(s) at time t
    P_0_t: float, # Initial discount factor to t
    P_0_T: float, # Initial discount factor to T
    a: float, 
    sigma: float
) -> np.ndarray:
    """
    Model-consistent bond price P(t, T) given HW state x_t.
    Vectorized over x_t (for Monte Carlo paths).
    """
    if t >= T_maturity:
        return np.ones_like(x_t)
        
    B = B_func(t, T_maturity, a)
    V = V_func(t, T_maturity, a, sigma)
    
    # Forward bond price
    fwd_P = P_0_T / P_0_t
    
    # HW adjustment
    exponent = -B * x_t - 0.5 * V
    
    return fwd_P * np.exp(exponent)

@njit
def simulate_xt_step(x_t: np.ndarray, dt: float, a: float, sigma: float, Z: np.ndarray) -> np.ndarray:
    """
    Exact simulation step for OU process x_t.
    dt: step size in years
    Z: standard normal random variables
    """
    if dt <= 0:
        return x_t.copy()
        
    mean_factor = np.exp(-a * dt)
    variance = (sigma**2 / (2 * a)) * (1.0 - np.exp(-2 * a * dt))
    std_dev = np.sqrt(max(0.0, variance))
    
    return x_t * mean_factor + std_dev * Z
