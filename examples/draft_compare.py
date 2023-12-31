from datetime import datetime

import numpy as np
import pandas as pd
from dotenv import load_dotenv
from tqdm import tqdm

from conmech.helpers import cmh
from conmech.helpers.config import Config, SimulationConfig
from conmech.scenarios import scenarios
from conmech.scenarios.scenarios import bunny_obstacles
from conmech.simulations import simulation_runner
from deep_conmech.graph.model_jax import RMSE
from deep_conmech.run_model import get_newest_checkpoint_path
from deep_conmech.training_config import TrainingConfig, get_train_config


def main():
    load_dotenv()
    cmh.print_jax_configuration()

    # for mode in ["skinning_backwards", "skinning", "net"]:  # Do not use normal
    #     run_simulation(mode)

    training_config = TrainingConfig(shell=False)
    checkpoint_path = get_newest_checkpoint_path(training_config)
    compare_latest(checkpoint_path.split("/")[-1])
    input("Press Enter to continue...")


def run_simulation(mode):
    # all_print_scenaros = scenarios.all_print(config.td, config.sc)
    # GraphModelDynamicJax.plot_all_scenarios(state, all_print_scenaros, training_config)

    config = get_train_config(shell=False, mode=mode)

    all_scenarios = scenarios.all_validation(config.td, config.sc)
    all_scenarios = all_scenarios[-1]

    simulation_runner.run_examples(
        all_scenarios=all_scenarios,
        file=__file__,
        plot_animation=False,
        config=Config(shell=False),
        save_all=True,
    )


def get_error(simulation_1, simulation_2, index, key):
    return RMSE(simulation_1[index][key], simulation_2[index][key])


def compare_latest(label=None):
    current_time: str = datetime.now().strftime("%m.%d-%H.%M.%S")
    input_path = "output"
    all_scene_files = cmh.find_files_by_extension(input_path, "scenes_comparer")

    dense = True
    path_id = "/scenarios/" if dense else "/scenarios_reduced/"
    scene_files = [f for f in all_scene_files if path_id in f]
    # all_arrays_path = max(scene_files, key=os.path.getctime)

    normal = cmh.get_simulation(scene_files, "skinning_backwards")
    skinning = cmh.get_simulation(scene_files, "skinning")
    net = cmh.get_simulation(scene_files, "net")

    simulation_len = min(len(normal), len(skinning), len(net))
    for key in [
        "norm_lifted_new_displacement",
        "recentered_norm_lifted_new_displacement",
        "displacement_old",
        "exact_acceleration",
        "normalized_nodes",
        "lifted_acceleration",
    ]:
        errors_skinning = []
        errors_net = []
        for index in tqdm(range(simulation_len)):
            errors_skinning.append(get_error(skinning, normal, index, key))
            errors_net.append(get_error(net, normal, index, key))

        print("Error net: ", np.mean(errors_net))
        print("Error skinning: ", np.mean(errors_skinning))
        all_errorrs = np.array([errors_skinning, errors_net])
        errors_df = pd.DataFrame(all_errorrs.T, columns=["skinning", "net"])
        plot = errors_df.plot()
        fig = plot.get_figure()
        fig.savefig(f"output/{current_time}_{label}_dense:{dense}_{key}.png")

    ####

    # dense = False
    # path_id = "/scenarios/" if dense else "/scenarios_reduced/"
    # scene_files = [f for f in scene_files if path_id in f]

    # skinning = cmh.get_simulation(all_scene_files, 'skinning')
    # net = cmh.get_simulation(all_scene_files, 'net')

    # simulation_len = min(len(normal), len(skinning), len(net))
    # for key in ['displacement_old', 'exact_acceleration', 'normalized_nodes']:
    #     errors_reduced = []
    #     for index in tqdm(range(simulation_len)):
    #         errors_reduced.append(get_error(skinning, net, index, key))

    #     errors_df = pd.DataFrame(np.array([errors_reduced]).T, columns=['reduced'])
    #     plot = errors_df.plot()
    #     fig = plot.get_figure()
    #     fig.savefig(f"output/{current_time}_{label}_dense:{dense}_errors_{key}.png")
    #     print(errors_df)


if __name__ == "__main__":
    main()
