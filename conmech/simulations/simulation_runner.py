import copy
import os
import time
from curses import noecho
from dataclasses import dataclass
from typing import Callable, Optional, Tuple

import numpy as np

from conmech.helpers import cmh, pkh
from conmech.helpers.config import Config
from conmech.plotting import plotter_2d, plotter_3d, plotter_common
from conmech.scenarios.scenarios import Scenario
from conmech.scene.scene import Scene
from conmech.scene.scene_temperature import SceneTemperature
from conmech.solvers.calculator import Calculator


def run_examples(
    all_scenarios,
    file,
    plot_animation,
    config: Config,
    simulate_dirty_data=False,
    get_scene_function: Optional[Callable] = None,
):
    for i, scenario in enumerate(all_scenarios):
        print(f"-----EXAMPLE {i + 1}/{len(all_scenarios)}-----")
        catalog = os.path.splitext(os.path.basename(file))[0].upper()
        run_scenario(
            solve_function=scenario.get_solve_function(),
            scenario=scenario,
            config=config,
            run_config=RunScenarioConfig(
                catalog=catalog,
                simulate_dirty_data=simulate_dirty_data,
                plot_animation=plot_animation,
            ),
            get_scene_function=get_scene_function,
        )
        print()
    print("DONE")


@dataclass
class RunScenarioConfig:
    catalog: Optional[str] = None
    simulate_dirty_data: bool = False
    compare_with_base_scene: bool = False
    plot_animation: bool = False
    save_all: bool = False


def run_scenario(
    solve_function: Callable,
    scenario: Scenario,
    config: Config,
    run_config: RunScenarioConfig,
    get_scene_function: Optional[Callable] = None,
) -> Tuple[Scene, str, float]:
    time_skip = config.print_skip
    ts = int(time_skip / scenario.time_step)
    index_skip = ts if run_config.save_all else 1
    plot_scenes_count = [0]

    save_files = True  # run_config.plot_animation or run_config.save_all
    if save_files:
        final_catalog = f"{config.output_catalog}/{config.current_time} - {run_config.catalog}"
        cmh.create_folders(f"{final_catalog}/scenarios")
        scenes_path = f"{final_catalog}/scenarios/{scenario.name}_DATA.scenes"
        if run_config.compare_with_base_scene:
            cmh.create_folders(f"{final_catalog}/scenarios_calculator")
            calculator_scenes_path = (
                f"{final_catalog}/scenarios_calculator/{scenario.name}_DATA.scenes"
            )
    else:
        final_catalog = ""
        scenes_path = ""
        calculator_scenes_path = ""

    def save_scene(scene: Scene, scenes_path: str, save_animation: bool):
        scene_copy = copy.copy(scene)
        scene_copy.prepare_to_save()

        arrays_path = scenes_path + "_data"
        nodes = scene.boundary_nodes
        elements = scene.boundaries.boundary_surfaces
        arrays = (nodes, elements)
        scenes_file, indices_file = pkh.open_files_append(arrays_path)
        with scenes_file, indices_file:
            pkh.append_data(data=arrays, data_path=arrays_path, lock=None)

        if save_animation:
            scenes_file, indices_file = pkh.open_files_append(scenes_path)
            with scenes_file, indices_file:
                pkh.append_data(data=scene_copy, data_path=scenes_path, lock=None)

    step = [0]  # TODO: #65 Clean

    def operation_save(scene: Scene, base_scene: Optional[Scene] = None):
        step[0] += 1
        plot_index = step[0] % ts == 0
        if run_config.save_all or plot_index:
            save_scene(
                scene=scene, scenes_path=scenes_path, save_animation=run_config.plot_animation
            )
            if base_scene is not None:
                save_scene(
                    scene=base_scene,
                    scenes_path=calculator_scenes_path,
                    save_animation=run_config.plot_animation,
                )
        if plot_index:
            plot_scenes_count[0] += 1

    normalize_by_rotation = False  #########################
    print(f"Creating scene... normalize: {normalize_by_rotation}")
    create_in_subprocess = False

    if get_scene_function is None:
        _get_scene_function = lambda randomize: scenario.get_scene(
            randomize=randomize,
            create_in_subprocess=create_in_subprocess,
            normalize_by_rotation=normalize_by_rotation,
        )
    else:
        _get_scene_function = lambda randomize: get_scene_function(
            config=config,
            scenario=scenario,
            randomize=randomize,
            create_in_subprocess=create_in_subprocess,
        )

    scene = _get_scene_function(randomize=run_config.simulate_dirty_data)
    if run_config.compare_with_base_scene:
        base_scene = _get_scene_function(randomize=False)
    else:
        base_scene = None

    fun_sim = lambda: simulate(
        scene=scene,
        base_scene=base_scene,
        solve_function=solve_function,
        scenario=scenario,
        simulate_dirty_data=run_config.simulate_dirty_data,
        compare_with_base_scene=run_config.compare_with_base_scene,
        config=config,
        operation=operation_save if save_files else None,
    )
    # cmh.profile(fun_sim)
    setting, energy_values = fun_sim()

    if run_config.plot_animation:
        animation_path = f"{final_catalog}/{scenario.name}.gif"
        plot_scenario_animation(
            scenario,
            config,
            animation_path,
            time_skip,
            index_skip,
            plot_scenes_count[0],
            all_scenes_path=scenes_path,
            all_calc_scenes_path=calculator_scenes_path
            if run_config.compare_with_base_scene
            else None,
        )

    return setting, scenes_path, energy_values


def plot_scenario_animation(
    scenario: Scenario,
    config: Config,
    animation_path: str,
    time_skip: float,
    index_skip: int,
    plot_scenes_count: int,
    all_scenes_path: str,
    all_calc_scenes_path: Optional[str],
):
    t_scale = plotter_common.get_t_scale(scenario, index_skip, plot_scenes_count, all_scenes_path)
    plot_function = (
        plotter_2d.plot_animation if scenario.dimension == 2 else plotter_3d.plot_animation
    )
    plot_function(
        save_path=animation_path,
        config=config,
        time_skip=time_skip,
        index_skip=index_skip,
        plot_scenes_count=plot_scenes_count,
        all_scenes_path=all_scenes_path,
        all_calc_scenes_path=all_calc_scenes_path,
        t_scale=t_scale,
    )


def prepare(scenario, scene: Scene, base_scene: Scene, current_time, with_temperature):
    forces = scenario.get_forces_by_function(scene, current_time)
    if with_temperature:
        heat = scenario.get_heat_by_function(scene, current_time)
        scene.prepare_tmp(forces, heat)
    else:
        scene.prepare(forces)

    if base_scene is not None:
        base_forces = scenario.get_forces_by_function(base_scene, current_time)
        base_scene.prepare(base_forces)


def simulate(
    scene,
    base_scene,
    solve_function,
    scenario: Scenario,
    simulate_dirty_data: bool,
    compare_with_base_scene: bool,
    config: Config,
    operation: Optional[Callable] = None,
) -> Tuple[Scene, float]:
    with_temperature = isinstance(scene, SceneTemperature)
    # reduced_scene = scene.all_layers[1].mesh
    # reduced_scene.normalize_and_set_obstacles(scenario.linear_obstacles, scenario.mesh_obstacles)

    solver_time = 0.0
    calculator_time = 0.0

    time_tqdm = scenario.get_tqdm(desc="Simulating", config=config)
    acceleration = None
    # reduced_acceleration = None
    temperature = None
    base_a = None
    energy_values = np.zeros(len(time_tqdm))
    for time_step in time_tqdm:
        current_time = (time_step + 1) * scene.time_step

        prepare(scenario, scene, base_scene, current_time, with_temperature)
        # prepare(scenario, reduced_scene, None, current_time, with_temperature)

        start_time = time.time()
        if with_temperature:
            acceleration, temperature = solve_function(
                scene, initial_a=acceleration, initial_t=temperature
            )
        else:
            acceleration = solve_function(scene, initial_a=acceleration)
            # reduced_scene.set_displacement
            # reduced_scene.interpolate_base(scene)
            # reduced_acceleration = solve_function(reduced_scene, initial_a=reduced_acceleration)
            # acceleration = scene.approximate_boundary_or_all_to_base(
            #     layer_number=1, reduced_values=reduced_acceleration
            # )
            # scene.approximate_boundary_or_all_from_base(
            #    layer_number=1, base_values=acceleration
            # )
        solver_time += time.time() - start_time

        if simulate_dirty_data:
            scene.make_dirty()

        if compare_with_base_scene:

            start_time = time.time()
            base_a = Calculator.solve(base_scene)  # TODO #65: save in setting
            calculator_time += time.time() - start_time

        if operation is not None:
            operation(scene, base_scene)  # (current_time, scene, base_scene, a, base_a)

        scene.iterate_self(acceleration, temperature=temperature)
        # reduced_scene.iterate_self(reduced_acceleration)

        if compare_with_base_scene:
            base_scene.iterate_self(base_a)

        # setting.remesh_self() # TODO #65

    comparison_str = f" | Calculator time: {calculator_time}" if compare_with_base_scene else ""
    print(f"    Solver time : {solver_time}{comparison_str}")
    print(f"MAX_K: {Calculator.MAX_K}")
    return scene, energy_values


def plot_setting(
    current_time,
    scene,
    path,
    base_scene,
    draw_detailed,
    extension,
):
    if scene.dimension == 2:
        fig = plotter_2d.get_fig()
        axs = plotter_2d.get_axs(fig)
        plotter_2d.plot_frame(
            fig=fig,
            axs=axs,
            scene=scene,
            current_time=current_time,
            draw_detailed=draw_detailed,
            base_scene=base_scene,
        )
        plotter_common.plt_save(path, extension)
    else:
        fig = plotter_3d.get_fig()
        axs = plotter_3d.get_axs(fig)
        plotter_3d.plot_frame(fig=fig, axs=axs, scene=scene, current_time=current_time)
        plotter_common.plt_save(path, extension)
