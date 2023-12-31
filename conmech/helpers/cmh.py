"""
conmech helpers
"""
import cProfile
import os
import pickle
import shutil
import sys
import time
from glob import glob
from pstats import Stats
from typing import Callable, Iterable

import psutil
from tqdm import tqdm

from conmech.helpers.config import Config


def get_from_os(name):
    return name in os.environ and int(os.environ[name])


def print_jax_configuration():
    if get_from_os("JAX_ENABLE_X64"):
        print("JAX 64 BIT MODE")
    else:
        print("JAX 32 BIT MODE")

    name = "JAX_PLATFORM_NAME"
    if name in os.environ:
        print(os.environ[name])


def get_used_memory_gb():
    return (
        psutil.Process(os.getpid()).memory_info().rss / 1024**3
    )  # (b -> kb -> mb -> gb)


def get_timestamp(config: Config):
    return int(time.time() * config.timestamp_skip)


def get_tqdm(iterable: Iterable, config: Config, desc=None, position=None) -> tqdm:
    return tqdm(iterable, desc=desc, position=position, ascii=config.shell)


def create_folder(path):
    if not os.path.exists(path):
        os.mkdir(path)


def create_folders(path):
    all_folders = path.split("/")
    final_path = ""
    for folder in all_folders:
        final_path += f"{folder}/"
        create_folder(final_path)


def get_all_contents(directory):
    return os.listdir(directory)


def clear_file(path):
    if os.path.exists(path):
        os.remove(path)


def clear_folder(directory):
    if not os.path.exists(directory):
        return
    shutil.rmtree(directory)


def recreate_folder(directory):
    clear_folder(directory)
    create_folder(directory)


def find_files_by_extension(directory, extension):
    files = []
    for dirpath, _, filenames in os.walk(directory):
        for filename in [f for f in filenames if f.endswith(f".{extension}")]:
            path = os.path.join(dirpath, filename)
            files.append(path)
    return files


def find_files_by_name(directory, name):
    files = []
    for dirpath, _, filenames in os.walk(directory):
        for filename in [f for f in filenames if name in f]:
            path = os.path.join(dirpath, filename)
            files.append(path)
    return files


def get_base_for_comarison():
    print("USING BASE FOR COMPARISON")
    all_paths = glob(
        "output/**/scenarios/*skinning_backwards*.scenes_comparer", recursive=True
    )
    assert len(all_paths) == 1
    return all_paths[0]


def get_run_label(config, scenario):
    return f"{config.current_time}_{scenario.simulation_config.mode}_{scenario.mesh_prop.mesh_type}_{scenario.name}"


def get_all_indices(data_path):
    all_indices = []
    try:
        with open(f"{data_path}_indices", "rb") as file:
            try:
                while True:
                    all_indices.append(pickle.load(file))
            except EOFError:
                pass
    except IOError:
        pass
    return all_indices


def load_simulation(simulation_path):
    all_indices = get_all_indices(simulation_path)
    simulation = []
    with open(simulation_path, "rb") as scenes_file:
        for byte_index in all_indices:
            scenes_file.seek(byte_index)
            data = pickle.load(scenes_file)
            simulation.append(data)
    return simulation


def get_simulation(scene_files, label):
    labels = [s for s in scene_files if label in s]
    # assert len(labels) == 1
    labels.sort()
    label = labels[-1]
    print(label)
    return load_simulation(label)


def get_exact_acceleration(scene, path):
    normal = load_simulation(path)
    exact_acceleration = normal[scene.step]["exact_acceleration"]
    scene.step += 1  # TODO: Move to iterate self

    reduced_exact_acceleration = scene.lift_acceleration_from_position(
        exact_acceleration
    )
    return exact_acceleration, reduced_exact_acceleration


####


def profile(function: Callable, baypass: bool = False):
    if baypass:
        return function()

    print("Profiling...")  # {function.__name__}...")
    pr = cProfile.Profile()
    pr.enable()
    result = function()
    pr.disable()
    stats = Stats(pr)
    stats.sort_stats("tottime").print_stats(20)  # "cumtime"

    # python -m scalene --cli --html --outfile output.html examples/examples_3d.py
    # from scalene import scalene_profiler
    # scalene_profiler.start()
    # result = function()
    # scalene_profiler.stop()

    # import jax
    # with jax.profiler.trace("./log", create_perfetto_link=False):
    #     result = function()

    return result


class DotDict(dict):
    """dot.notation access to dictionary attributes"""

    __getattr__ = dict.get
    __setattr__ = dict.__setitem__
    __delattr__ = dict.__delitem__


class Console:
    HEADER = "\033[95m"
    OKBLUE = "\033[94m"
    OKCYAN = "\033[96m"
    OKGREEN = "\033[92m"
    WARNING = "\033[93m"
    FAIL = "\033[91m"
    ENDC = "\033[0m"
    BOLD = "\033[1m"
    UNDERLINE = "\033[4m"

    @staticmethod
    def print_warning(text):
        print(f"{Console.WARNING}{text}{Console.ENDC}")

    @staticmethod
    def print_fail(text):
        print(f"{Console.FAIL}{text}{Console.ENDC}")


class HiddenPrints:
    def __init__(self):
        self._original_stdout = sys.stdout

    def __enter__(self):
        sys.stdout = open(os.devnull, "w", encoding="utf-8")

    def __exit__(self, exc_type, exc_val, exc_tb):
        sys.stdout.close()
        sys.stdout = self._original_stdout
