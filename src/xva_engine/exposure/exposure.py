import numpy as np

def calculate_exposures(V_ns: np.ndarray, C: np.ndarray):
    """
    Computes pathwise and expected exposures.
    V_ns: shape (steps, paths)
    C: shape (steps, paths)
    """
    # Pathwise exposures
    # E(t,w) = max(V(t,w) - C(t,w), 0)
    E = np.maximum(V_ns - C, 0.0)
    
    # NE(t,w) = max(C(t,w) - V(t,w), 0)  [Negative Exposure from counterparty perspective = bank's exposure to them]
    # For DVA, we want our exposure to our default => max(C - V, 0)
    NE = np.maximum(C - V_ns, 0.0)
    
    # Expected Exposures
    EE = np.mean(E, axis=1)
    ENE = np.mean(NE, axis=1)
    
    return {
        "E_paths": E,
        "NE_paths": NE,
        "EE": EE,
        "ENE": ENE
    }
