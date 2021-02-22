"""
Created at 18.02.2021
"""

import scipy.optimize
import numpy as np
from simulation.solvers.solver import Solver
from simulation.solvers.solver_methods import make_L2


class Optimization(Solver):

    def __init__(self, grid, inner_forces, outer_forces, mu_coef, lambda_coef, contact_law, friction_bound):
        super().__init__(grid, inner_forces, outer_forces, mu_coef, lambda_coef, contact_law, friction_bound)
        self.loss = make_L2(jn=contact_law.potential_normal_direction)

    @property
    def C(self):
        raise NotImplementedError()

    @property
    def E(self):
        raise NotImplementedError()

    def solve(self, initial_guess: np.ndarray) -> np.ndarray:
        result = scipy.optimize.minimize(
            self.loss,
            initial_guess,
            args=(self.grid.indNumber(), self.grid.BorderEdgesC, self.grid.Edges, self.grid.Points, self.C, self.E),
            method='BFGS',
            options={'disp': True, 'maxiter': len(initial_guess) * 1e5},
            tol=1e-12
        ).x
        return result
