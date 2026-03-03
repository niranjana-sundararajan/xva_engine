import numpy as np
from ..io.schemas import CSA

def calculate_collateral(V_ns: np.ndarray, csa: CSA) -> np.ndarray:
    """
    Computes collateral C(t, omega) given the netting set value V_ns.
    V_ns: shape (num_grid_points, num_paths)
    """
    if csa is None or csa.mode == "none":
        return np.zeros_like(V_ns)
        
    if csa.mode == "perfect_vm":
        return V_ns.copy()
        
    if csa.mode == "threshold":
        # Simplified threshold logic:
        # If V_ns > threshold, C = V_ns - threshold
        # If V_ns < -threshold, C = V_ns + threshold
        # Else C = 0
        C = np.zeros_like(V_ns)
        C[V_ns > csa.threshold] = V_ns[V_ns > csa.threshold] - csa.threshold
        C[V_ns < -csa.threshold] = V_ns[V_ns < -csa.threshold] + csa.threshold
        return C

    return np.zeros_like(V_ns)  # pragma: no cover
