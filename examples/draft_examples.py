from functools import partial

import numpy as np
from dotenv import load_dotenv

from conmech.helpers import cmh
from conmech.helpers.config import Config, SimulationConfig
from conmech.properties.body_properties import TimeDependentBodyProperties
from conmech.properties.mesh_properties import MeshProperties
from conmech.properties.obstacle_properties import ObstacleProperties
from conmech.properties.schedule import Schedule
from conmech.scenarios.scenarios import (
    M_ARMADILLO_3D,
    M_BALL_3D,
    M_BUNNY_3D,
    M_CUBE_3D,
    M_TWIST_3D,
    Scenario,
    all_train,
    bunny_fall_3d,
    bunny_obstacles,
    bunny_rotate_3d,
    default_body_prop,
    default_obstacle_prop,
    f_rotate_3d,
)
from conmech.simulations import simulation_runner
from conmech.state.obstacle import Obstacle


def main():
    load_dotenv()
    cmh.print_jax_configuration()

    def get_simulation_config(mode="normal", use_pca=False):
        return SimulationConfig(
            use_normalization=False,
            use_linear_solver=False,
            use_green_strain=True,
            use_nonconvex_friction_law=False,
            use_constant_contact_integral=False,
            use_lhs_preconditioner=False,
            with_self_collisions=True,
            use_pca=use_pca,
            mesh_layer_proportion=4,
            mode=mode,
        )

    final_time = 4
    scale_forces = 5.0

    # all_print_scenaros = scenarios.all_print(config.td, config.sc)
    # GraphModelDynamicJax.plot_all_scenarios(state, all_print_scenaros, training_config)

    all_scenarios = [
        # bunny_fall_3d(
        #     mesh_density=32,
        #     scale=1,
        #     final_time=final_time,
        #     simulation_config=get_simulation_config(mode),
        #     scale_forces=scale_forces,
        # ),
        # bunny_obstacles(
        #     mesh_density=32,
        #     scale=1,
        #     final_time=final_time,
        #     simulation_config=get_simulation_config(),
        #     scale_forces=scale_forces,
        # ),
        # bunny_rotate_3d(
        #     mesh_density=32,
        #     scale=1,
        #     final_time=final_time,
        #     simulation_config=get_simulation_config(mode),
        #     scale_forces=scale_forces,
        # ),
        # Scenario(
        #     name="bunny_fall",
        #     mesh_prop=MeshProperties(
        #         dimension=3,
        #         mesh_type=M_BUNNY_3D,
        #         scale=[1],
        #         mesh_density=[32],
        #     ),
        #     body_prop=TimeDependentBodyProperties(
        #         mu=12.0,
        #         lambda_=12.0,
        #         theta=4.0,
        #         zeta=4.0,
        #         mass_density=1.0,
        #     ),
        #     schedule=Schedule(final_time=2),
        #     forces_function=np.array([0.0, 0.0, -1.0]),
        #     obstacle=Obstacle(  # 0.3
        #         np.array([[[0.0, 0.7, 1.0]], [[1.0, 1.0, 0.0]]]),
        #         ObstacleProperties(hardness=100.0, friction=5.0),
        #     ),
        #     simulation_config=simulation_config,
        # ),
        # Scenario(
        #     name="bunny_fall",
        #     mesh_prop=MeshProperties(
        #         dimension=3,
        #         mesh_type=M_BUNNY_3D,
        #         scale=[1],
        #         mesh_density=[32],
        #     ),
        #     body_prop=TimeDependentBodyProperties(
        #         mu=8,
        #         lambda_=8,
        #         theta=8,
        #         zeta=8,
        #         mass_density=1.0,
        #     ),
        #     schedule=Schedule(final_time=2, time_step=0.01),
        #     forces_function=np.array([0.0, 0.0, -1.0]),
        #     obstacle=Obstacle(
        #         None,
        #         ObstacleProperties(hardness=100.0, friction=5.0),
        #         all_mesh=[
        #             MeshProperties(
        #                 dimension=3,
        #                 mesh_type="slide_left",
        #                 scale=[1],
        #                 mesh_density=[16],
        #                 initial_position=[0, 0, -0.5],
        #             ),
        #         ],  # x,y,z front,right,bottom
        #     ),
        #     simulation_config=simulation_config,
        # ),
        # Scenario(
        #     name="bunny_roll",
        #     mesh_prop=MeshProperties(
        #         dimension=3,
        #         mesh_type=M_BUNNY_3D,
        #         scale=[1],
        #         mesh_density=[16],  # 8
        #     ),
        #     body_prop=TimeDependentBodyProperties(
        #         mu=12,
        #         lambda_=12,
        #         theta=16,
        #         zeta=16,
        #         mass_density=1.0,
        #     ),
        #     schedule=Schedule(final_time=final_time),
        #     forces_function=f_rotate_3d,
        #     obstacle=Obstacle(
        #         np.array([[[0.0, 0.0, 1.0]], [[0.0, 0.0, 0.3]]]), default_obstacle_prop
        #     ),
        # )
        Scenario(
            name="armadillo_fall",
            mesh_prop=MeshProperties(
                dimension=3,
                mesh_type=M_ARMADILLO_3D,
                scale=[1],
                mesh_density=[16],
            ),
            body_prop=TimeDependentBodyProperties(
                mu=16 * 4,
                lambda_=16 * 4,
                theta=8 * 4,
                zeta=8 * 4,
                mass_density=1.0,
            ),
            schedule=Schedule(final_time=final_time, time_step=0.01),
            forces_function=scale_forces * np.array([0.0, 0.0, -1.0]),
            obstacle=Obstacle(
                geometry=None,  # np.array([[[0.7, 0.0, 1.0]], [[1.0, 1.0, 0.0]]]),
                properties=ObstacleProperties(
                    hardness=200.0, friction=4.0
                ),  # friction=0.1
                all_mesh=[
                    MeshProperties(
                        dimension=3,
                        mesh_type="slide_up",
                        scale=[1],
                        mesh_density=[16],
                        initial_position=[0, 0, -1],  # -2],
                    )
                ],
            ),
            simulation_config=get_simulation_config(),
        ),
    ]

    simulation_runner.run_examples(
        all_scenarios=all_scenarios,
        file=__file__,
        plot_animation=True,
        config=Config(shell=False, animation_backend="three"),
        save_all=False,
    )


if __name__ == "__main__":
    main()
