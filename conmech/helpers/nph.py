"""
numpy helpers
"""
from ctypes import ArgumentError

import numba
import numpy as np
from scipy import sparse


def stack(data):
    return data.T.flatten()


def stack_column(data):
    return data.T.flatten().reshape(-1, 1)


stack_column_numba = numba.njit(stack_column)


def unstack(vector, dim):
    return vector.reshape(-1, dim, order="F")


def unstack_and_sum_columns(data, dim, keepdims=False):
    return np.sum(unstack(data, dim), axis=1, keepdims=keepdims)


def to_sparse(data):
    return None if data is None else sparse.csr_matrix(data)


def elementwise_dot(matrix_1, matrix_2, keepdims=False):
    return (matrix_1 * matrix_2).sum(axis=1, keepdims=keepdims)


def get_occurances(data):
    return np.array(list(set(data.flatten())))


def close_modulo(value, divider):
    if divider is None:
        return True
    return np.allclose(value % divider, 0.0) or np.allclose(value % divider, divider)


def euclidean_norm(vector, keepdims=False):
    data = (vector**2).sum(axis=-1, keepdims=keepdims)
    if isinstance(vector, np.ndarray):
        return np.sqrt(data)
    return data.sqrt()
    # return np.linalg.norm(vector, axis=-1)
    # return np.sqrt(np.sum(vector ** 2, axis=-1))[..., np.newaxis]


@numba.njit
def euclidean_norm_numba(vector):
    data = (vector**2).sum(axis=-1)
    return np.sqrt(data)


@numba.njit
def normalize_euclidean_numba(data):
    norm = euclidean_norm_numba(data)
    reshaped_norm = norm if data.ndim == 1 else norm.reshape(-1, 1)
    return data / reshaped_norm


def get_normal(vector, normal):
    return elementwise_dot(vector, normal, keepdims=True)


def get_normal_tangential(vector, normal):
    normal_vector = get_normal(vector, normal)
    tangential_vector = vector - (normal_vector * normal)
    return normal_vector, tangential_vector


def get_tangential(vector, normal):
    _, tangential_vector = get_normal_tangential(vector, normal)
    return tangential_vector


@numba.njit
def get_tangential_numba(vector, normal):
    normal_vector = vector @ normal
    tangential_vector = vector - (normal_vector * normal)
    return tangential_vector


def get_tangential_2d(normal):
    return np.array((normal[..., 1], -normal[..., 0])).T


def complete_base(base_seed, closest_seed_index=0):
    dim = base_seed.shape[-1]
    # normalized_base_seed = normalize_euclidean_numba(base_seed)
    if dim == 2:
        base = orthonormalize(base_seed)
    elif dim == 3:
        rolled_base_seed = np.roll(base_seed, -closest_seed_index, axis=0)
        rolled_base = orthonormalize(rolled_base_seed)
        base = np.roll(rolled_base, closest_seed_index, axis=0)
    else:
        raise ArgumentError
    # base = normalize_euclidean_numba(unnormalized_base)
    return base


def generate_base(dimension):
    while True:
        vectors = generate_normal(rows=dimension, columns=dimension, sigma=1)
        try:
            base = orthonormalize(vectors)
            return base
        except:
            print("Base generation error")


def correct_base(base):
    dim = len(base)
    for i in range(dim):
        for j in range(i + 1, dim):
            if not np.allclose(base[i] @ base[j], 0):
                return False

    if not np.allclose(euclidean_norm(base), np.ones(dim)):
        return False

    if len(base) == 2 and not np.allclose(np.cross(*base), 1):
        return False
    if len(base) == 3 and not np.allclose(np.cross(*base[:2]), base[2]):
        return False
    return True


def orthonormalize(vectors):
    vectors = normalize_euclidean_numba(vectors)
    base = np.linalg.qr(vectors)[0]
    if len(base) == 2:
        base[0] *= -1.0  # keep right orientetion
    if correct_base(base):
        return base
    raise ArgumentError


def get_in_base(vectors, base):
    return vectors @ base.T


@numba.njit
def len_x_numba(corners):
    return corners[2] - corners[0]


@numba.njit
def len_y_numba(corners):
    return corners[3] - corners[1]


@numba.njit
def min_numba(corners):
    return [corners[0], corners[1]]


@numba.njit
def max_numba(corners):
    return [corners[2], corners[3]]


@numba.njit
def get_node_index_numba(node, nodes):
    for i, n in enumerate(nodes):
        if np.sum(np.abs(node - n)) < 0.0001:
            return i
    raise ArgumentError


def generate_normal(rows, columns, sigma):
    return np.random.normal(loc=0.0, scale=sigma, size=[rows, columns])


def generate_uniform_circle(rows, columns, low, high):
    result = generate_normal(rows=rows, columns=columns, sigma=1.0)
    normalized_result = normalize_euclidean_numba(result)
    radius = np.random.uniform(low=low, high=high, size=[rows, 1])
    return radius * normalized_result


def append_euclidean_norm(data):
    return np.hstack((data, euclidean_norm(data, keepdims=True)))


@numba.njit(inline="always")
def length(p_1, p_2):
    return np.sqrt((p_1[0] - p_2[0]) ** 2 + (p_1[1] - p_2[1]) ** 2)


# @numba.njit
# def calculate_angle_numba(new_up_vector):
#     old_up_vector = np.array([0., 1.])
#     angle = (2 * (new_up_vector[0] >= 0) - 1) * np.arccos(np.dot(new_up_vector, old_up_vector))
#     return angle
#
# @numba.njit
# def rotate_numba(vectors, angle):
#     s = np.sin(angle)
#     c = np.cos(angle)
#
#     rotated_vectors = np.zeros_like(vectors)
#     rotated_vectors[:, 0] = vectors[:, 0] * c - vectors[:, 1] * s
#     rotated_vectors[:, 1] = vectors[:, 0] * s + vectors[:, 1] * c
#
#     return rotated_vectors
