"""
Created at 21.08.2019
"""

import numpy as np
import pytest
from dataclasses import dataclass

from conmech.problem_solver import Quasistatic as QuasistaticProblem
from conmech.problems import Quasistatic
from examples.p_slope_contact_law import make_slope_contact_law
from tests.examples_regression.std_boundary import standard_boundary_nodes


@pytest.fixture(params=[  # TODO #28
    # "global optimization",  # TODO #29
    "schur"
])
def solving_method(request):
    return request.param


def generate_test_suits():
    test_suites = []

    # Simple example

    @dataclass()
    class QuasistaticSetup(Quasistatic):
        grid_height: ... = 1
        cells_number: ... = (2, 5)
        mu_coef: ... = 4
        lambda_coef: ... = 4
        th_coef: ... = 4
        ze_coef: ... = 4
        time_step: ... = 0.1
        contact_law: ... = make_slope_contact_law(slope=1e1)

        @staticmethod
        def inner_forces(x, y):
            return np.array([-0.2, -0.2])

        @staticmethod
        def outer_forces(x, y):
            return np.array([0, 0])

        @staticmethod
        def friction_bound(u_nu):
            return 0

        @staticmethod
        def is_contact(x, y):
            return y == 0

        @staticmethod
        def is_dirichlet(x, y):
            return x == 0

    setup_m02_m02 = QuasistaticSetup()

    expected_displacement_vector_m02_m02 = \
        [[0., 0.],
         [0.03174709, 0.02791856],
         [0.04874281, 0.04364196],
         [0.05728231, 0.05630736],
         [0.06046744, 0.06519966],
         [0.0593477, 0.07272406],
         [0.05109885, 0.08033477],
         [0.04084442, 0.08284004],
         [0.03832706, 0.07139745],
         [0.03054836, 0.05772677],
         [0.01850842, 0.03964177],
         [0.00648465, 0.01742879],
         [0., 0.],
         [0., 0.]]

    test_suites.append((setup_m02_m02, expected_displacement_vector_m02_m02))

    # p = 0 and opposite forces

    setup_0_02_p_0 = QuasistaticSetup()
    setup_0_02_p_0.contact_law = make_slope_contact_law(slope=0)

    def inner_forces(x, y):
        return np.array([0, 0.2])
    setup_0_02_p_0.inner_forces = inner_forces

    expected_displacement_vector_0_02_p_0 = \
        [[0., 0.],
         [-0.11383076, -0.120908],
         [-0.18658829, -0.30784272],
         [-0.2237752, -0.54111482],
         [-0.23790945, -0.78856463],
         [-0.24043737, -1.03329948],
         [0.00000004, -1.03311517],
         [0.24043746, -1.03329948],
         [0.23790954, -0.78856465],
         [0.22377527, -0.54111483],
         [0.18658834, -0.30784273],
         [0.11383078, -0.12090801],
         [0., 0.],
         [0., 0.]]

    test_suites.append((setup_0_02_p_0, expected_displacement_vector_0_02_p_0))

    # p = 0

    setup_0_m02_p_0 = QuasistaticSetup()
    setup_0_m02_p_0.contact_law = make_slope_contact_law(slope=0)

    def inner_forces(x, y):
        return np.array([0, -0.2])
    setup_0_m02_p_0.inner_forces = inner_forces

    expected_displacement_vector_0_m02_p_0 = [[-v[0], -v[1]]
                                              for v in expected_displacement_vector_0_02_p_0]
    test_suites.append((setup_0_m02_p_0, expected_displacement_vector_0_m02_p_0))

    # various changes

    @dataclass()
    class QuasistaticSetup(Quasistatic):
        grid_height: ... = 1.37
        cells_number: ... = (2, 5)
        mu_coef: ... = 4.58
        lambda_coef: ... = 3.33
        th_coef: ... = 2.11
        ze_coef: ... = 4.99
        time_step: ... = 0.1
        contact_law: ... = make_slope_contact_law(slope=2.71)

        @staticmethod
        def inner_forces(x, y):
            return np.array([0, -0.2])

        @staticmethod
        def outer_forces(x, y):
            return np.array([0.3, 0.0])

        @staticmethod
        def friction_bound(u_nu):
            return 0.0

        @staticmethod
        def is_contact(x, y):
            return y == 0

        @staticmethod
        def is_dirichlet(x, y):
            return x == 0

    setup_var = QuasistaticSetup()
    expected_displacement_vector_var = \
        [[0., 0.],
         [0.0260751, 0.0532828],
         [0.03693136, 0.15587502],
         [0.03541443, 0.2943425],
         [0.02747103, 0.45490576],
         [0.01291648, 0.62761593],
         [-0.16050501, 0.64941692],
         [-0.35617262, 0.66290617],
         [-0.32973288, 0.48007344],
         [-0.28372655, 0.31781761],
         [-0.21878207, 0.17832828],
         [-0.12898926, 0.07212696],
         [0., 0.],
         [0., 0.]]

    test_suites.append((setup_var, expected_displacement_vector_var))

    return test_suites


@pytest.mark.parametrize('setup, expected_displacement_vector', generate_test_suits())
def test_global_optimization_solver(solving_method, setup, expected_displacement_vector):
    runner = QuasistaticProblem(setup, solving_method)
    results = runner.solve(n_steps=32)

    displacement = results[-1].mesh.initial_points[:] - results[-1].displaced_points[:]
    std_ids = standard_boundary_nodes(runner.mesh.initial_points, runner.mesh.cells)

    # print result
    np.set_printoptions(precision=8, suppress=True)
    print(repr(displacement[std_ids]))

    np.testing.assert_array_almost_equal(
        displacement[std_ids], expected_displacement_vector, decimal=3)