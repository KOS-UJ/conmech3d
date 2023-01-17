import os
import pickle
from io import BufferedReader

import jax
import jax.numpy as jnp

from conmech.helpers import nph


def find_files_by_extension(directory, extension):
    files = []
    for dirpath, _, filenames in os.walk(directory):
        for filename in [f for f in filenames if f.endswith(f".{extension}")]:
            path = os.path.join(dirpath, filename)
            files.append(path)
    return files


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


def open_file_read(path: str):
    return open(path, "rb")


def load_byte_index(byte_index: int, data_file: BufferedReader):
    data_file.seek(byte_index)
    data = pickle.load(data_file)
    return data


def get_scenes():
    input_path = "/home/michal/Desktop/conmech/output"
    scene_files = find_files_by_extension(input_path, "scenes")  # scenes_data
    path_id = "/scenarios/"
    scene_files = [f for f in scene_files if path_id in f]
    all_arrays_path = max(scene_files, key=os.path.getctime)
    all_arrays_name = os.path.basename(all_arrays_path).split("DATA")[0]

    print(f"FILE: {all_arrays_name}")

    all_indices = get_all_indices(all_arrays_path)
    scenes = []
    scenes_file = open_file_read(all_arrays_path)
    with scenes_file:
        for byte_index in all_indices:
            scene = load_byte_index(
                byte_index=byte_index,
                data_file=scenes_file,
            )
            scenes.append(scene)
    return scenes


def get_projection(data, latent_dim=2000):
    projection_mean = data.mean(axis=0)  # columnwise mean = 0
    svd = jax.numpy.linalg.svd(data - projection_mean, full_matrices=False)
    # (svd[0] @ jnp.diag(svd[1]) @ svd[2])
    projection_matrix = svd[2][:latent_dim].T
    return {"matrix": projection_matrix, "mean": projection_mean.reshape(-1, 1)}


def project_to_latent(projection, data_stack):
    data_stack_zeroed = data_stack - projection["mean"]
    latent = projection["matrix"].T @ data_stack_zeroed
    return latent


def project_from_latent(projection, latent):
    data_stack_zeroed = projection["matrix"] @ latent
    data_stack = data_stack_zeroed + projection["mean"]
    return data_stack


def save_pca(projection, file_path="./output/PCA"):
    with open(file_path, "wb") as file:
        pickle.dump(projection, file)


def load_pca(file_path="./output/PCA"):
    with open(file_path, "rb") as file:
        projection = pickle.load(file)
    return projection


def get_data(scenes):
    data_list = []
    count = len(scenes)
    for scene in scenes:
        u = jnp.array(scene.exact_acceleration)
        u_stack = nph.stack_column(u)
        data_list.append(u_stack)

    data = jnp.array(data_list).reshape(count, -1)
    return data, u_stack, u


def run():
    scenes = get_scenes()
    data, u_stack, u = get_data(scenes)

    original_projection = get_projection(data)
    save_pca(original_projection)
    projection = load_pca()

    latent = project_to_latent(projection, u_stack)

    u_reprojected_stack = project_from_latent(projection, latent)
    u_reprojected = nph.unstack(u_reprojected_stack, dim=3)

    print("Error max: ", jnp.abs(u_reprojected - u).max())
    return 0