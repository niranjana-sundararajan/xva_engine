import numpy as np
from ..models.hw1f import simulate_xt_step

class MonteCarloEngine:
    def __init__(self, num_paths: int, seed: int, hw_a: float, hw_sigma: float):
        self.num_paths = num_paths
        self.seed = seed
        self.hw_a = hw_a
        self.hw_sigma = hw_sigma

    def simulate_paths(self, grid_days: np.ndarray) -> np.ndarray:
        """
        Simulates HW 1F state variable x_t over the grid.
        Returns array of shape (num_grid_points, num_paths).
        """
        rng = np.random.default_rng(self.seed)
        num_steps = len(grid_days)
        
        x_paths = np.zeros((num_steps, self.num_paths), dtype=np.float64)
        
        for i in range(1, num_steps):
            dt_days = grid_days[i] - grid_days[i-1]
            dt_years = dt_days / 365.0
            
            Z = rng.standard_normal(self.num_paths)
            x_paths[i, :] = simulate_xt_step(
                x_t=x_paths[i-1, :], 
                dt=dt_years, 
                a=self.hw_a, 
                sigma=self.hw_sigma, 
                Z=Z
            )

        return x_paths
