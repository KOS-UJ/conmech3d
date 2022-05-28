from dataclasses import dataclass
from typing import Optional

import psutil
from torch import nn

from conmech.helpers.config import Config

TEST = False
DIMENSION = 2


@dataclass
class TrainingData:
    dimension: int = DIMENSION

    train_scale: int = 1
    validation_scale: int = 1
    print_scale: int = 1

    dataset: str = "synthetic"  # synthetic # calculator
    final_time: float = 0.5 if TEST else 8
    mesh_density: int = 64  # 64 if dimension == 2 else 16
    adaptive_training_mesh_scale: Optional[float] = 0.8

    forces_random_scale: float = 4.0
    obstacle_origin_max_scale: float = 3.0 * train_scale  # less
    obstacle_origin_min_scale: float = 0.4 * train_scale  # 2.9
    initial_corners_scale: float = 0.1
    displacement_random_scale: float = 0.2  # 0.2
    velocity_random_scale: float = 2.5

    zero_forces_proportion: float = 0.2
    zero_displacement_proportion: float = 0.2
    zero_velocity_proportion: float = 0.2
    corners_scale_proportion: float = 0.8

    displacement_to_velocity_noise: float = 0.1
    displacement_in_random_factor: float = 0.005 * displacement_random_scale
    velocity_in_random_factor: float = 0.005 * velocity_random_scale

    save_at_minutes: int = 10
    validate_at_epochs: Optional[int] = 1  # 10
    validate_scenarios_at_epochs: Optional[int] = None  # 3  # 30

    use_energy_as_loss: bool = True
    batch_size: int = 16  # 128  # 16  # 128
    valid_batch_size: int = batch_size  # 128
    synthetic_batches_in_epoch: int = 4  # 32  # 1 if TEST else 256  # 512

    use_dataset_statistics: bool = False
    input_batch_norm: bool = True
    internal_batch_norm: bool = False
    layer_norm: bool = True

    dropout_rate: Optional[float] = None  # 0.0  # 0.1 # 0.2  0.05
    skip_connections: bool = True
    gradient_clip = 10.0  # None

    attention_heads_count: Optional[int] = None  # 5  # None 1 3 5

    initial_learning_rate: float = 1e-3  # 1e-3  # 1e-4 # 1e-5
    learning_rate_decay: float = 1.0  # 0.995
    final_learning_rate: float = 1e-6

    activation = nn.ReLU()  # nn.PReLU() LeakyReLU
    latent_dimension: int = 128
    encoder_layers_count: int = 0
    processor_layers_count: int = 0
    decoder_layers_count: int = 0
    mesh_layers_count: int = 3
    message_passes: int = 3


@dataclass
class TrainingConfig(Config):
    td: TrainingData = TrainingData()
    device: str = "_"

    dataloader_workers = 4
    synthetic_generation_workers = 1  # 2

    total_mempry_gb = psutil.virtual_memory().total / 1024**3
    total_memory_limit_gb = round(total_mempry_gb * 0.9, 2)
    synthetic_generation_memory_limit_gb = round(
        (total_mempry_gb * 0.8) / synthetic_generation_workers, 2
    )

    dataset_images_count: Optional[float] = None  # 100

    load_train_features_to_ram: bool = True
    load_train_targets_to_ram: bool = False
    log_dataset_stats: bool = True
    with_train_scenes_file: bool = True

    compare_with_base_scene = False
    max_epoch_number: Optional[int] = None
    datasets_main_path: str = "datasets"
    log_catalog: str = "log"
    load_newest_train: bool = False  # True
