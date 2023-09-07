# from conmech.helpers.config import SET_ENV
# if name == "__main__":
#     SET_ENV()

from dotenv import load_dotenv

load_dotenv()

import argparse
import os
from argparse import ArgumentParser, Namespace
from ctypes import ArgumentError
from pathlib import Path

import jax
import numpy as np
import torch
import torch.distributed as dist
import torch.multiprocessing

from conmech.helpers import cmh, pca
from conmech.helpers.config import Config, SimulationConfig
from conmech.scenarios import scenarios
from conmech.scenarios.scenarios import bunny_fall_3d
from conmech.simulations import simulation_runner
from conmech.solvers.calculator import Calculator
from deep_conmech.data import base_dataset
from deep_conmech.data.calculator_dataset import CalculatorDataset
from deep_conmech.data.synthetic_dataset import SyntheticDataset
from deep_conmech.graph.model_jax import GraphModelDynamicJax, save_tf_model
from deep_conmech.graph.net_jax import CustomGraphNetJax
from deep_conmech.helpers import dch
from deep_conmech.training_config import TrainingConfig, TrainingData, get_train_config


def setup_distributed(rank: int, world_size: int):
    os.environ["MASTER_ADDR"] = "localhost"
    # with socketserver.TCPServer(("localhost", 0), None) as s:
    #     free_port = str(s.server_address[1])
    free_port = "12348"
    os.environ["MASTER_PORT"] = free_port
    # os.environ["TORCH_DISTRIBUTED_DEBUG"] = "DETAIL"
    dist.init_process_group("nccl", rank=rank, world_size=world_size)


def cleanup_distributed():
    dist.destroy_process_group()


def get_device_count(config):
    return len(jax.local_devices())


def initialize_data(config: TrainingConfig):
    device_count = get_device_count(config)

    train_dataset = get_train_dataset(
        config.td.dataset, config=config, device_count=device_count
    )
    train_dataset.initialize_data()

    all_validation_datasets = get_all_val_datasets(
        config=config, rank=0, world_size=1, device_count=device_count  # 1
    )
    for datasets in all_validation_datasets:
        datasets.initialize_data()

    return train_dataset, all_validation_datasets


def train(config: TrainingConfig):
    train_dataset, all_validation_datasets = initialize_data(config=config)

    train_single(
        config,
        train_dataset=train_dataset,
        all_validation_datasets=all_validation_datasets,
    )


def dist_run(
    rank: int,
    world_size: int,
    config: TrainingConfig,
):
    setup_distributed(rank=rank, world_size=world_size)
    train_single(config, rank=rank, world_size=world_size)
    cleanup_distributed()


def train_single(
    config, rank=0, world_size=1, train_dataset=None, all_validation_datasets=None
):
    device_count = get_device_count(config)
    if train_dataset is None:
        train_dataset = get_train_dataset(
            config.td.dataset,
            config=config,
            rank=rank,
            world_size=world_size,
            device_count=device_count,
        )
        train_dataset.load_indices()

    statistics = (
        train_dataset.get_statistics() if config.td.use_dataset_statistics else None
    )
    if config.td.use_dataset_statistics:
        train_dataset.statistics = statistics

    if all_validation_datasets is None:
        all_validation_datasets = get_all_val_datasets(
            config=config, rank=rank, world_size=world_size, device_count=device_count
        )

    all_print_datasets = scenarios.all_print(config.td, config.sc)
    for dataset in all_validation_datasets:
        if config.td.use_dataset_statistics:
            dataset.statistics = statistics

    model = GraphModelDynamicJax(
        train_dataset=train_dataset,
        all_validation_datasets=all_validation_datasets,
        print_scenarios=all_print_datasets,
        config=config,
        statistics=statistics,
    )

    if config.load_newest_train:
        model.load_checkpoint(path=checkpoint_path)
    model.train()


def visualize(config: TrainingConfig):
    import netron

    checkpoint_path = get_newest_checkpoint_path(config)
    dataset = get_train_dataset(config.td.dataset, config=config)
    dataset.initialize_data()

    model_path = "log/jax_model.tflite"
    state = GraphModelDynamicJax.load_checkpointed_net(path=checkpoint_path)
    save_tf_model(model_path, state, dataset)

    netron.start(model_path)


def plot(config: TrainingConfig):
    if config.td.use_dataset_statistics:
        train_dataset = get_train_dataset(config.td.dataset, config=config)
        statistics = train_dataset.get_statistics()
    else:
        statistics = None
    all_print_scenaros = scenarios.all_print(config.td, config.sc)

    checkpoint_path = get_newest_checkpoint_path(config)
    state = GraphModelDynamicJax.load_checkpointed_net(path=checkpoint_path)
    GraphModelDynamicJax.plot_all_scenarios(state, all_print_scenaros, config)


def run_pca(config: TrainingConfig):
    mesh_density = 32  # 16 #32
    final_time = 2.0  # 8.0  # 2.0
    simulation_config = SimulationConfig(
        use_normalization=False,
        use_linear_solver=False,
        use_green_strain=True,
        use_nonconvex_friction_law=False,
        use_constant_contact_integral=False,
        use_lhs_preconditioner=False,
        with_self_collisions=True,
        mode="pca",
    )
    all_scenarios = [
        scenarios.bunny_fall_3d(
            mesh_density=mesh_density,
            scale=1,
            final_time=final_time,
            simulation_config=simulation_config,
        ),
        scenarios.bunny_rotate_3d(
            mesh_density=mesh_density,
            scale=1,
            final_time=final_time,
            simulation_config=simulation_config,
        ),
    ]

    dataset = get_train_dataset(
        dataset_type=config.td.dataset, config=config, device_count=1
    )
    # datasets = get_all_val_datasets(config=config, rank=0, world_size=1, device_count=1)
    # dataset = datasets[1]
    dataset.initialize_data()
    dataloader = base_dataset.get_train_dataloader(dataset)

    # dataloader = None

    pca.run(dataloader, latent_dim=200, scenario=all_scenarios[0])

    simulation_runner.run_examples(
        all_scenarios=all_scenarios,
        file=__file__,
        plot_animation=True,
        config=Config(shell=False),
        save_all=True,
    )


def get_train_dataset(
    dataset_type,
    config: TrainingConfig,
    rank: int = 0,
    world_size: int = 1,
    device_count=None,
    item_fn=None,
):
    if device_count is None:
        device_count = get_device_count(config)
    if dataset_type == "synthetic":
        train_dataset = SyntheticDataset(
            description="train",
            load_data_to_ram=config.load_training_data_to_ram,
            with_scenes_file=config.with_train_scenes_file,
            randomize=True,
            config=config,
            rank=rank,
            world_size=world_size,
            device_count=device_count,
            item_fn=item_fn,
        )
    elif dataset_type == "calculator":
        train_dataset = CalculatorDataset(
            description="train",
            all_scenarios=scenarios.all_train(config.td, config.sc),
            load_data_to_ram=config.load_training_data_to_ram,
            with_scenes_file=config.with_train_scenes_file,
            randomize=True,
            config=config,
            rank=rank,
            world_size=world_size,
            device_count=device_count,
            item_fn=item_fn,
        )
    else:
        raise ValueError("Wrong dataset type")
    return train_dataset


def get_all_val_datasets(
    config: TrainingConfig, rank: int, world_size: int, device_count: int
):
    all_val_datasets = []
    for all_scenarios in scenarios.all_validation(config.td, config.sc):
        description = "validation_" + str.join(
            "/", [scenario.name for scenario in all_scenarios]
        )
        all_val_datasets.append(
            CalculatorDataset(
                description=description,
                all_scenarios=all_scenarios,
                load_data_to_ram=config.load_validation_data_to_ram,
                with_scenes_file=False,
                randomize=False,
                config=config,
                rank=rank,
                world_size=world_size,
                device_count=device_count,
            )
        )
    return all_val_datasets


def get_newest_checkpoint_path_jax(config: TrainingConfig):
    def get_index_jax(path):
        return int(path.split("/")[-2].split(" ")[0])

    all_checkpoint_paths = cmh.find_files_by_name(config.output_catalog, "checkpoint")
    if not all_checkpoint_paths:
        raise ArgumentError("No saved models")
    newest_index = np.argmax(
        np.array([get_index_jax(path) for path in all_checkpoint_paths])
    )

    path = str(Path(all_checkpoint_paths[newest_index]).parent.absolute())
    print(f"============================ Taking saved model {path.split('/')[-1]}")
    return path


def get_newest_checkpoint_path(config: TrainingConfig):
    return get_newest_checkpoint_path_jax(config)


def main(args: Namespace):
    cmh.print_jax_configuration()
    print(f"MODE: {args.mode}, PID: {os.getpid()}")
    # dch.cuda_launch_blocking()
    # torch.autograd.set_detect_anomaly(True)
    # print(numba.cuda.gpus)

    config = get_train_config(shell=args.shell, mode="normal")

    # dch.set_torch_sharing_strategy()
    dch.set_memory_limit(config=config)
    print(f"Running using {config.device}")

    if args.mode == "train":
        train(config)
    if args.mode == "profile":
        config.max_epoch_number = 2
        train(config)
    if args.mode == "plot":
        plot(config)
    if args.mode == "visualize":
        visualize(config)
    if args.mode == "pca":
        run_pca(config)


if __name__ == "__main__":
    # torch.multiprocessing.set_start_method("spawn")  # forkserver")
    parser = ArgumentParser()
    parser.add_argument(
        "--mode",
        type=str,
        choices=["train", "plot", "profile", "visualize", "pca"],
        default="plot",
        help="Running mode of aplication",
    )
    parser.add_argument(
        "--shell", action=argparse.BooleanOptionalAction, default=False
    )  # Python 3.9+
    args = parser.parse_args()
    # with jax.disable_jit():
    main(args)
