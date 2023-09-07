from datetime import datetime

import numpy as np
import pandas as pd
from dotenv import load_dotenv
from tqdm import tqdm

from conmech.helpers import cmh
from conmech.helpers.config import Config
from conmech.scenarios import scenarios
from conmech.simulations import simulation_runner
from deep_conmech.graph.model_jax import RMSE
from deep_conmech.run_model import get_newest_checkpoint_path
from deep_conmech.training_config import TrainingConfig, get_train_config


def main():
    load_dotenv()
    modes = [
        "skinning_backwards",
        "skinning",
        "net",
        "pca",
    ]  # , "pca"]  # Do not use "normal" - does not contain reduced
    # run_all_simulations(modes=modes)
    compare_latest(modes=modes)
    input("Press Enter to continue...")


def run_all_simulations(modes):
    cmh.print_jax_configuration()

    for mode in modes:
        run_simulation(mode)


def run_simulation(mode):
    # all_print_scenaros = scenarios.all_print(config.td, config.sc)
    # GraphModelDynamicJax.plot_all_scenarios(state, all_print_scenaros, training_config)

    config = get_train_config(shell=False, mode=mode)

    all_scenarios = scenarios.all_validation(config.td, config.sc)
    all_scenarios = all_scenarios[-1]

    # all_scenarios[0].schedule.final_time = 1.

    simulation_runner.run_examples(
        all_scenarios=all_scenarios,
        file=__file__,
        plot_animation=False,
        config=Config(shell=False),
        save_all=True,
    )


def get_error(simulation_1, simulation_2, index, key):
    return RMSE(simulation_1[index][key], simulation_2[index][key])


def compare_latest(modes):
    training_config = TrainingConfig(shell=False)
    checkpoint_path = get_newest_checkpoint_path(training_config)
    input_path = "output"
    all_scene_files = cmh.find_files_by_extension(input_path, "scenes_comparer")
    current_time: str = datetime.now().strftime("%m.%d-%H.%M.%S")
    label = checkpoint_path.split("/")[-1]
    main_path = f"output/{current_time}_COMPARE_{label}"

    dense = True
    path_id = "/scenarios/" if dense else "/scenarios_reduced/"
    scene_files = [f for f in all_scene_files if path_id in f]
    # all_arrays_path = max(scene_files, key=os.path.getctime)

    def add_centered(simulation):
        for step in simulation:
            step["centered_new_displacement"] = step["new_displacement"] - step[
                "new_displacement"
            ].mean(axis=0)
        return simulation

    base = cmh.get_simulation(scene_files, "skinning_backwards")
    # base = add_centered(base)

    cmh.create_folder(main_path)
    for key in [
        "norm_lifted_new_displacement",
        "recentered_norm_lifted_new_displacement",
        "normalized_nodes",
        # "centered_new_displacement",
        # "new_displacement",
        # "displacement_old",
        # "exact_acceleration",
        # "normalized_nodes",
        # # "lifted_acceleration",
    ]:
        errors_df = pd.DataFrame()
        for mode in modes:
            if mode == "skinning_backwards":
                continue
            pretendent = cmh.get_simulation(scene_files, mode)
            # pretendent = add_centered(pretendent)

            simulation_len = min(len(base), len(pretendent))

            errors = []
            for index in tqdm(range(simulation_len)):
                errors.append(get_error(base, pretendent, index=index, key=key))
            errors_df[mode] = np.array(errors)
            print(f"Error {mode} {key}: ", np.mean(errors))
            print()

        plot = errors_df.plot()
        fig = plot.get_figure()
        fig.savefig(f"{main_path}/{dense}_{key}.png")


if __name__ == "__main__":
    main()
