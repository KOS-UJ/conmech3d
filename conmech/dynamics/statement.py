from dataclasses import dataclass
from typing import Optional

import numpy as np


@dataclass
class Variables:
    displacement: Optional[np.ndarray] = None
    velocity: Optional[np.ndarray] = None
    temperature: Optional[np.ndarray] = None
    time_step: Optional[float] = None


class Statement:
    def __init__(self, dynamics):
        self.dynamics = dynamics
        self.left_hand_side = None
        self.right_hand_side = None

    def update_left_hand_side(self, var: Variables):
        raise NotImplementedError()

    def update_right_hand_side(self, var: Variables):
        raise NotImplementedError()

    def update(self, var: Variables):
        self.update_left_hand_side(var)
        self.update_right_hand_side(var)


class StaticStatement(Statement):
    def update_left_hand_side(self, var: Variables):
        self.left_hand_side = self.dynamics.elasticity

    def update_right_hand_side(self, var: Variables):
        self.right_hand_side = self.dynamics.forces.forces_vector


class QuasistaticStatement(Statement):
    def update_left_hand_side(self, var: Variables):
        self.left_hand_side = self.dynamics.viscosity

    def update_right_hand_side(self, var: Variables):
        assert var.displacement is not None

        self.right_hand_side = (
            self.dynamics.forces.forces_vector - self.dynamics.elasticity @ var.displacement.T
        )


class DynamicStatement(Statement):
    def update_left_hand_side(self, var):
        assert var.time_step is not None

        self.left_hand_side = (
            self.dynamics.viscosity + (1 / var.time_step) * self.dynamics.acceleration_operator
        )

    def update_right_hand_side(self, var):
        assert var.displacement is not None
        assert var.velocity is not None
        assert var.time_step is not None
        assert var.temperature is not None

        A = -1 * self.dynamics.elasticity @ var.displacement

        A += (1 / var.time_step) * self.dynamics.acceleration_operator @ var.velocity

        A += self.dynamics.thermal_expansion.T @ var.temperature

        self.right_hand_side = self.dynamics.forces.forces_vector + A


class TemperatureStatement(Statement):
    def update_left_hand_side(self, var):
        assert var.time_step is not None

        ind = self.dynamics.independent_nodes_count

        self.left_hand_side = (1 / var.time_step) * self.dynamics.acceleration_operator[
            :ind, :ind
        ] + self.dynamics.thermal_conductivity[:ind, :ind]

    def update_right_hand_side(self, var):
        assert var.velocity is not None
        assert var.time_step is not None
        assert var.temperature is not None

        rhs = (-1) * self.dynamics.thermal_expansion @ var.velocity

        ind = self.dynamics.independent_nodes_count

        rhs += ((1 / var.time_step)
                * self.dynamics.acceleration_operator[:ind, :ind] @ var.temperature)
        self.right_hand_side = rhs
        # self.right_hand_side = self.inner_temperature.F[:, 0] + Q1 - C2Xv - C2Yv  # TODO #50
