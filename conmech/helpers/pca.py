import os
import pickle
from io import BufferedReader

import jax
import jax.numpy as jnp
import numpy as np
from tqdm import tqdm

from conmech.helpers import cmh, lnh, nph
from conmech.properties.mesh_properties import MeshProperties
from conmech.scene.scene import Scene
from conmech.simulations.simulation_runner import create_scene


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
    input_path = "/home/michal/Desktop/conmech3d/output"
    scene_files = cmh.find_files_by_extension(input_path, "scenes")  # scenes_data
    path_id = "/scenarios/"
    scene_files = [f for f in scene_files if path_id in f and "SAVED" not in f]

    assert len(scene_files) == 1
    # all_arrays_path = max(scene_files, key=os.path.getctime)
    scenes = []
    for all_arrays_path in scene_files:
        if "SAVED" in all_arrays_path:
            continue
        all_arrays_name = os.path.basename(all_arrays_path).split("DATA")[0]
        print(f"FILE: {all_arrays_name}")

        all_indices = get_all_indices(all_arrays_path)
        scenes_file = open_file_read(all_arrays_path)
        with scenes_file:
            for byte_index in all_indices:
                scene = load_byte_index(
                    byte_index=byte_index,
                    data_file=scenes_file,
                )
                scenes.append(scene)
    return scenes


def save_pca(projection, file_path="./output/PCA"):
    with open(file_path, "wb") as file:
        pickle.dump(projection, file)


def load_pca(file_path="./output/PCA"):
    with open(file_path, "rb") as file:
        projection = pickle.load(file)
    return projection


def get_displacement_new(scene):
    velocity = scene.velocity_old + scene.time_step * scene.exact_acceleration
    displacement = scene.displacement_old + scene.time_step * velocity
    return displacement


def get_data_scenes(scenes):
    data_list = []
    for scene in tqdm(scenes):
        # print(scene.moved_base)
        u = scene.get_lifted_displacement()
        u_stack = nph.stack(u)
        data_list.append(u_stack)

    data = jnp.array(data_list)
    return data, u_stack, u


def get_data_dataset(dataloader, scene):
    data_list = []
    count = 3000
    print(f"LIMIT TO {count}")
    for i, sample in enumerate(tqdm(dataloader)):  # check randomness
        target = sample[0][1]

        original_displacement = jnp.array(target["new_displacement"])

        original_rotation = scene.get_rotation(original_displacement)
        random_rotation = jnp.linalg.qr(np.random.rand(3, 3))[0]
        new_rotation = original_rotation.T @ random_rotation

        moved_nodes = scene.initial_nodes + original_displacement
        displacement_mean = np.mean(moved_nodes, axis=0)
        rotated_moved_nodes = lnh.get_in_base2(
            (moved_nodes - displacement_mean), new_rotation
        )
        displacement = rotated_moved_nodes - scene.initial_nodes
        displacement += displacement_mean  # 20 * np.random.rand(3) ###

        displacement_stack = nph.stack(displacement)
        data_list.append(displacement_stack)
        if i > count:
            break

    data = jnp.array(
        data_list
    )  # Sort by displacement and get max, plot hist # np.linalg.norm(data_list[190])
    return data, displacement_stack, displacement


def get_projection(data, latent_dim):
    projection_mean = 0  # data.mean(axis=0)

    svd = jax.numpy.linalg.svd(data - projection_mean, full_matrices=False)
    # (svd[0] @ jnp.diag(svd[1]) @ svd[2])

    projection_matrix = svd[2][:latent_dim]
    # projection_matrix = jax.experimental.sparse.eye(data.shape[1])

    return {"matrix": projection_matrix, "mean": projection_mean}


def project_to_latent(projection, data):
    data_zeroed = data - projection["mean"]
    latent = projection["matrix"] @ data_zeroed
    return latent


def project_from_latent(projection, latent):
    data_stack_zeroed = projection["matrix"].T @ latent
    data_stack = data_stack_zeroed + projection["mean"]
    return data_stack


def p_to_vector(projection, data):
    return project_to_latent(projection, nph.stack(data))


def p_from_vector(projection, latent):
    return nph.unstack(project_from_latent(projection, latent), dim=3)


def run(dataloader, latent_dim, scenario):
    if dataloader is None:
        scenes = get_scenes()
        data, sample_u_stack, sample_u = get_data_scenes(scenes)
    else:
        scene = create_scene(scenario)
        data, sample_u_stack, sample_u = get_data_dataset(
            dataloader=dataloader, scene=scene
        )

    original_projection = get_projection(data, latent_dim)
    save_pca(original_projection)

    # projection = load_pca()
    # latent = project_to_latent(projection, sample_u_stack)
    # u_reprojected_stack = project_from_latent(projection, latent)
    # u_reprojected = nph.unstack(u_reprojected_stack, dim=3)
    # print("Error max: ", jnp.abs(u_reprojected - sample_u).max())
    return 0
