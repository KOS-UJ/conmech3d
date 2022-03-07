from ast import If

import torch
from deep_conmech.common import *
from deep_conmech.graph.helpers import thh
from deep_conmech.graph.setting.setting_randomized import *
from deep_conmech.graph.setting.setting_randomized import L2_normalized_cuda
from deep_conmech.simulator.setting.setting_forces import *
from torch_geometric.data import Data


def obstacle_resistance_potential_normal_torch(normal_displacement):
    value = config.OBSTACLE_HARDNESS * 2.0 * normal_displacement ** 2
    return (normal_displacement > 0) * value * ((1.0 / config.TIMESTEP) ** 2)


def obstacle_resistance_potential_tangential_torch(
    normal_displacement, tangential_velocity
):
    value = config.OBSTACLE_FRICTION * thh.euclidean_norm_torch(tangential_velocity)
    return (normal_displacement > 0) * value * ((1.0 / config.TIMESTEP))


def integrate_torch(
    nodes,
    v,
    faces,
    closest_to_faces_obstacle_normals,
    closest_to_faces_obstacle_origins,
):
    normals = -closest_to_faces_obstacle_normals

    edge_node = nodes[faces]
    edge_v = v[faces]

    middle_node = torch.mean(edge_node, axis=1)
    middle_v = torch.mean(edge_v, axis=1)

    middle_node_normal = nph.elementwise_dot(
        middle_node - closest_to_faces_obstacle_origins, normals
    )
    middle_v_normal = nph.elementwise_dot(middle_v, normals, keepdims=True)

    middle_v_tangential = middle_v - (middle_v_normal * normals)

    edge_lengths = thh.euclidean_norm_torch(edge_node[:, 0] - edge_node[:, 1])
    resistance_normal = obstacle_resistance_potential_normal_torch(middle_node_normal)
    resistance_tangential = obstacle_resistance_potential_tangential_torch(
        middle_node_normal, middle_v_tangential
    )
    result = torch.sum(edge_lengths * (resistance_normal + resistance_tangential))
    return result


def L2_normalized_obstacle_correction_cuda(
    cleaned_normalized_a,
    C,
    normalized_E,
    normalized_boundary_v_old,
    normalized_boundary_points,
    boundary_faces,
    normalized_closest_to_faces_obstacle_normals,
    normalized_closest_to_faces_obstacle_origins,
    normalized_a_correction=None,
):
    if normalized_a_correction is not None:
        normalized_a = cleaned_normalized_a - normalized_a_correction
    else:
        normalized_a = cleaned_normalized_a

    internal = L2_normalized_cuda(normalized_a, C, normalized_E)

    boundary_nodes_count = normalized_boundary_v_old.shape[0]
    normalized_boundary_a = normalized_a[:boundary_nodes_count, :]

    normalized_boundary_v_new = (
        normalized_boundary_v_old + config.TIMESTEP * normalized_boundary_a
    )
    normalized_boundary_points_new = (
        normalized_boundary_points + config.TIMESTEP * normalized_boundary_v_new
    )

    boundary_integral = integrate_torch(
        normalized_boundary_points_new,
        normalized_boundary_v_new,
        boundary_faces,
        normalized_closest_to_faces_obstacle_normals,
        normalized_closest_to_faces_obstacle_origins,
    )

    return internal + boundary_integral


#################################


@njit
def set_diff(data, position, row, i, j):
    vector = data[j] - data[i]
    row[position : position + 2] = vector
    row[position + 2] = np.linalg.norm(vector)


@njit  # (parallel=True)
def get_edges_data(
    edges,
    initial_nodes,
    u_old,
    v_old,
    forces,
    boundary_faces_count,
    obstacle_normal,
    boundary_centers_penetration_scale,
):  # , forces
    edges_number = edges.shape[0]
    edges_data = np.zeros((edges_number, config.EDGE_DATA_DIM))
    for e in range(edges_number):
        i = edges[e, 0]
        j = edges[e, 1]

        set_diff(initial_nodes, 0, edges_data[e], i, j)
        set_diff(u_old, 3, edges_data[e], i, j)
        set_diff(v_old, 6, edges_data[e], i, j)
        set_diff(forces, 9, edges_data[e], i, j)
        """#TODO: move to points
        if e < boundary_faces_count:
            penetration = boundary_centers_penetration_scale[e].item()
            if penetration > 0:
                edges_data[e, 12:14] = obstacle_normal
                edges_data[e, 14] = penetration
            edges_data[e, 15] = 1.0
        """
    return edges_data


###################################3


def L2_obstacle_nvt(
    boundary_a,
    C_boundary,
    E_boundary,
    boundary_v_old,
    boundary_points,
    boundary_faces,
    closest_to_faces_obstacle_normals,
    closest_to_faces_obstacle_origins,
):  # np via torch

    value_torch = L2_normalized_obstacle_correction_cuda(
        thh.to_torch_double(boundary_a).to(thh.device),
        thh.to_torch_double(C_boundary).to(thh.device),
        thh.to_torch_double(E_boundary).to(thh.device),
        thh.to_torch_double(boundary_v_old).to(thh.device),
        thh.to_torch_double(boundary_points).to(thh.device),
        thh.to_torch_long(boundary_faces).to(thh.device),
        thh.to_torch_double(closest_to_faces_obstacle_normals).to(thh.device),
        thh.to_torch_double(closest_to_faces_obstacle_origins).to(thh.device),
        None,
    )
    value = thh.to_np_double(value_torch)
    return value  # .item()


class SettingInput(SettingRandomized):
    def __init__(
        self,
        mesh_type,
        mesh_density_x,
        mesh_density_y,
        scale_x,
        scale_y,
        is_adaptive,
        create_in_subprocess,
    ):
        super().__init__(
            mesh_type,
            mesh_density_x,
            mesh_density_y,
            scale_x,
            scale_y,
            is_adaptive,
            create_in_subprocess,
        )

    def get_edges_data_torch(self, edges):
        edges_data = get_edges_data(
            edges,
            self.normalized_initial_nodes,
            self.input_u_old,
            self.input_v_old,
            self.input_forces,
            self.boundary_faces_count,
            self.normalized_obstacle_normal,
            self.boundary_centers_penetration_scale,
        )
        return thh.to_torch_double(edges_data)

    @property
    def x(self):
        # data = torch.ones(self.nodes_count, 1)
        data = torch.hstack(
            (
                thh.get_data_with_euclidean_norm(self.input_forces_torch),
                # thh.get_data_with_euclidean_norm(self.input_u_old_torch),
                # thh.get_data_with_euclidean_norm(self.input_v_old_torch)
            )
        )
        return data

    def get_data(self, setting_index=None, exact_normalized_a_torch=None):
        # edge_index_torch, edge_attr = remove_self_loops(
        #    self.contiguous_edges_torch, self.edges_data_torch
        # )
        # Do not use "face" in name (probably reserved in PyG)
        directional_edges = np.vstack((self.edges, np.flip(self.edges, axis=1)))
        data = Data(
            pos=thh.set_precision(self.normalized_initial_nodes_torch),
            x=thh.set_precision(self.x),
            edge_index=thh.get_contiguous_torch(directional_edges),
            edge_attr=thh.set_precision(self.get_edges_data_torch(directional_edges)),
            reshaped_C=self.C_torch.reshape(-1, 1),
            normalized_E=self.normalized_E_torch,
            normalized_a_correction=self.normalized_a_correction_torch,
            setting_index=setting_index,
            exact_normalized_a=exact_normalized_a_torch,
            normalized_boundary_v_old=self.normalized_boundary_v_old_torch,
            normalized_closest_to_fac_obstacle_normals=self.normalized_closest_to_faces_obstacle_normals_torch,
            normalized_closest_to_fac_obstacle_origins=self.normalized_closest_to_faces_obstacle_origins_torch,
            boundary_nodes_count=self.boundary_nodes_count_torch,
            normalized_boundary_points=self.normalized_boundary_points_torch,
            boundary_fac_count=self.boundary_faces_count_torch,
            boundary_fac=self.boundary_faces_torch,
            # pin_memory=True,
            # num_workers=1
        )
        """
        transform = T.Compose(
            [
                T.TargetIndegree(norm=False),
                T.Cartesian(norm=False),
                T.Polar(norm=False),
            ]  # add custom for multiple 'pos' types
        )  # T.OneHotDegree(),
        transform(data)
        """
        return data

    def normalized_L2_obstacle_nvt(self, normalized_boundary_a_vector):
        return L2_obstacle_nvt(
            thh.unstack(normalized_boundary_a_vector),
            self.C_boundary,
            self.normalized_E_boundary,
            self.normalized_boundary_v_old,
            self.normalized_boundary_points,
            self.boundary_faces,
            self.normalized_closest_to_faces_obstacle_normals,
            self.normalized_closest_to_faces_obstacle_origins,
        )

