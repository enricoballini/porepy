import sys
import os
import pdb
import pickle
import numpy as np
import scipy as sp
import torch
import postprocess_utils as pu
from concurrent.futures import ProcessPoolExecutor, ThreadPoolExecutor

sys.path.append("../../mypythonmodulescut")
sys.path.append("../../../mypythonmodulescut")

pp_path = "../porepy/src"
if pp_path not in sys.path:
    sys.path.append(pp_path)

import porepy as pp


import nnrom  # weirdly required by torch.load

os.system("clear")


data_folder = "/media/inspiron/DATI Toshiba/here/data"

data_folder_root = data_folder + ""

test_dataset_id = np.loadtxt(data_folder_root + "/test_dataset_id", dtype=np.int32)


def compute_fault_traction_fom(idx_mu):
    """ """
    save_folder = "./results/mech/" + str(idx_mu)

    friction_coeff = 0.45  # form spe paper

    with open(data_folder + "/mech/800/sd_fract.pkl", "rb") as fle:
        sd_fract = pickle.load(fle)

    exporter = pp.Exporter(
        sd_fract,
        file_name="sd_fract",
        folder_name=save_folder,
    )

    u_ref = np.load(
        data_folder + "/mech/" + str(idx_mu) + "/displacement_" + str(0.0) + ".npy"
    )
    u_b_filled_ref = np.load(
        data_folder
        + "/mech/"
        + str(idx_mu)
        + "/displacement_boundary_"
        + str(0.0)
        + ".npy"
    )

    times_mech = np.loadtxt(data_folder + "/TIMES_MECH")

    def parallel_fcn(time):
        """ """
        time = time[0]
        print("time = ", time)

        stress_tensor_grad = sp.sparse.load_npz(
            data_folder
            + "/mech/"
            + str(idx_mu)
            + "/stress_tensor_grad_"
            + str(time)
            + ".npz"
        )
        bound_stress = sp.sparse.load_npz(data_folder + "/mech/800/bound_stress.npz")

        u = np.load(
            data_folder + "/mech/" + str(idx_mu) + "/displacement_" + str(time) + ".npy"
        )

        u_b_filled = np.load(
            data_folder
            + "/mech/"
            + str(idx_mu)
            + "/displacement_boundary_"
            + str(time)
            + ".npy"
        )

        dim = 3
        fracture_faces_id = np.load(data_folder + "/mech/800/fracture_faces_id.npy")
        normal_area = np.load(data_folder + "/mech/800/normal_area.npy")
        face_area = np.load(data_folder + "/mech/800/face_area.npy")

        normal = normal_area / np.linalg.norm(normal_area, ord=2)

        delta_u = u - u_ref
        delta_u_b_filled = u_b_filled - u_b_filled_ref

        # reference: -------------------
        T_ref = stress_tensor_grad * u + bound_stress * u_b_filled
        T_vect_ref = np.reshape(T_ref, (dim, -1), order="F")
        T_vect_frac_ref = T_vect_ref[:, fracture_faces_id]

        T_area_ref = (
            stress_tensor_grad * u_ref + bound_stress * u_b_filled_ref
        )  # this is in Newton
        T_vect_ref = (
            np.reshape(T_area_ref, (dim, -1), order="F") / face_area
        )  # this in in Pascal
        T_vect_frac_ref = T_vect_ref[:, fracture_faces_id]

        normal_projection = pp.map_geometry.normal_matrix(normal=normal)
        tangential_projetion = pp.map_geometry.tangent_matrix(normal=normal)

        T_normal_ref = normal_projection @ T_vect_frac_ref
        T_normal_normal_ref = np.dot(normal, T_normal_ref)
        T_normal_norm_ref = np.linalg.norm(T_normal_ref.T, ord=2, axis=1)
        T_tangential_ref = tangential_projetion @ T_vect_frac_ref
        T_tangential_y_ref = T_tangential_ref[
            1
        ].T  # one on-plane direction is aligned with y
        T_tangential_t_ref = np.linalg.norm(
            np.array([T_tangential_ref[0], T_tangential_ref[2]]).T, ord=2, axis=1
        )  # the other tangential dir, t, is made of x and z
        T_tangential_norm_ref = np.linalg.norm(T_tangential_ref.T, ord=2, axis=1)

        # deltas: ------------------------------------
        delta_T_area = (
            stress_tensor_grad * delta_u + bound_stress * delta_u_b_filled
        )  # this is in Newton
        delta_T_vect = (
            np.reshape(delta_T_area, (dim, -1), order="F") / face_area
        )  # this in in Pascal
        delta_T_vect_frac = delta_T_vect[:, fracture_faces_id]

        delta_T_normal = normal_projection @ delta_T_vect_frac
        delta_T_normal_normal = np.dot(normal, delta_T_normal)
        delta_T_normal_norm = np.linalg.norm(delta_T_normal.T, ord=2, axis=1)
        delta_T_tangential = tangential_projetion @ delta_T_vect_frac
        delta_T_tangential_y = delta_T_tangential[
            1
        ].T  # one on-plane direction is aligned with y
        delta_T_tangential_t = np.linalg.norm(
            np.array([delta_T_tangential[0], delta_T_tangential[2]]).T, ord=2, axis=1
        )  # the other tangential dir, t, is made of x and z
        delta_T_tangential_norm = np.linalg.norm(delta_T_tangential.T, ord=2, axis=1)

        delta_cff = (
            delta_T_tangential_norm
            + T_tangential_norm_ref
            - friction_coeff * (delta_T_normal_normal + T_normal_normal_ref)
        ) - (
            T_tangential_norm_ref - friction_coeff * T_normal_normal_ref
        )  # this is in Pascal

        np.save(save_folder + "/cff_" + str(time), delta_cff)

        delta_traction_tangential_norm = delta_T_tangential_norm * (
            face_area[fracture_faces_id]
        )  # in newton
        delta_traction_normal_normal = delta_T_normal_normal * (
            face_area[fracture_faces_id]
        )

        exporter.write_vtu(
            [
                (sd_fract, "T_xyz", delta_T_vect_frac),
                (sd_fract, "T_out_of_plane", delta_T_normal_norm),
                (sd_fract, "T_out_of_plane_signed", delta_T_normal_normal),
                (sd_fract, "minus_T_out_of_plane_signed", -delta_T_normal_normal),
                (sd_fract, "T_tangential_norm", delta_T_tangential_norm),
                (sd_fract, "T_tangential_y", delta_T_tangential_y),
                (sd_fract, "T_tangential_t", delta_T_tangential_t),
                (sd_fract, "cff", delta_cff),
                (sd_fract, "T_normal_normal_ref", T_normal_normal_ref),  # added
                (
                    sd_fract,
                    "T_tangential_norm_ref",
                    T_tangential_norm_ref,
                ),  # added
                (
                    sd_fract,
                    "delta_traction_tangential_norm",
                    delta_traction_tangential_norm,
                ),  # added
                (
                    sd_fract,
                    "minus_delta_traction_normal",
                    -delta_traction_normal_normal,
                ),  # added
            ],
            time_dependent=True,
            time_step=time,
        )

    arguments = np.array_split(times_mech, 3)
    with ThreadPoolExecutor() as executor:
        output_executor = executor.map(parallel_fcn, arguments)

    # pay attention, executor doesnt print niether stops at errors! you have to force it printing the output

    for i in output_executor:
        print(i)

    print("Traction computed!\n")


def compute_fault_traction_nn_cut(idx_mu):
    """
    I do not save/upload all the data bcs I do not have enough space
    """
    save_folder = "./results/nn/" + str(idx_mu)

    friction_coeff = 0.45  # form spe paper

    with open(data_folder + "/mech/800/sd_fract.pkl", "rb") as fle:
        sd_fract = pickle.load(fle)

    exporter = pp.Exporter(
        sd_fract,
        file_name="sd_fract",
        folder_name=save_folder,
    )

    u_ref = np.load(
        data_folder + "/mech/" + str(idx_mu) + "/displacement_" + str(0.0) + ".npy"
    )
    u_b_filled_ref = np.load(
        data_folder
        + "/mech/"
        + str(idx_mu)
        + "/displacement_boundary_"
        + str(0.0)
        + ".npy"
    )

    times_mech = np.loadtxt(data_folder + "/TIMES_MECH")

    def parallel_fcn(time):
        """
        - TODO: some variables are read outside the parallel_fcn scope, imporve it
        """
        time = time[0]
        print("time = ", time)

        # FOM: -------------------------------------------------------------------------
        stress_tensor_grad = sp.sparse.load_npz(
            data_folder
            + "/mech/"
            + str(idx_mu)
            + "/stress_tensor_grad_"
            + str(time)
            + ".npz"
        )
        bound_stress = sp.sparse.load_npz(data_folder + "/mech/800/bound_stress.npz")

        u = np.load(
            data_folder + "/mech/" + str(idx_mu) + "/displacement_" + str(time) + ".npy"
        )

        u_b_filled = np.load(
            data_folder
            + "/mech/"
            + str(idx_mu)
            + "/displacement_boundary_"
            + str(time)
            + ".npy"
        )

        dim = 3
        fracture_faces_id = np.load(data_folder + "/mech/800/fracture_faces_id.npy")
        normal_area = np.load(data_folder + "/mech/800/normal_area.npy")
        face_area = np.load(data_folder + "/mech/800/face_area.npy")

        normal = normal_area / np.linalg.norm(normal_area, ord=2)

        delta_u = u - u_ref
        delta_u_b_filled = u_b_filled - u_b_filled_ref

        # reference: -------------------
        T_ref = stress_tensor_grad * u + bound_stress * u_b_filled
        T_vect_ref = np.reshape(T_ref, (dim, -1), order="F")
        T_vect_frac_ref = T_vect_ref[:, fracture_faces_id]

        T_area_ref = (
            stress_tensor_grad * u_ref + bound_stress * u_b_filled_ref
        )  # this is in Newton
        T_vect_ref = (
            np.reshape(T_area_ref, (dim, -1), order="F") / face_area
        )  # this in in Pascal
        T_vect_frac_ref = T_vect_ref[:, fracture_faces_id]

        normal_projection = pp.map_geometry.normal_matrix(normal=normal)
        tangential_projetion = pp.map_geometry.tangent_matrix(normal=normal)

        T_normal_ref = normal_projection @ T_vect_frac_ref
        T_normal_normal_ref = np.dot(normal, T_normal_ref)
        T_normal_norm_ref = np.linalg.norm(T_normal_ref.T, ord=2, axis=1)
        T_tangential_ref = tangential_projetion @ T_vect_frac_ref
        T_tangential_y_ref = T_tangential_ref[
            1
        ].T  # one on-plane direction is aligned with y
        T_tangential_t_ref = np.linalg.norm(
            np.array([T_tangential_ref[0], T_tangential_ref[2]]).T, ord=2, axis=1
        )  # the other tangential dir, t, is made of x and z
        T_tangential_norm_ref = np.linalg.norm(T_tangential_ref.T, ord=2, axis=1)

        # deltas: ------------------------------------
        delta_T_area = (
            stress_tensor_grad * delta_u + bound_stress * delta_u_b_filled
        )  # this is in Newton
        delta_T_vect = (
            np.reshape(delta_T_area, (dim, -1), order="F") / face_area
        )  # this in in Pascal
        delta_T_vect_frac = delta_T_vect[:, fracture_faces_id]

        delta_T_normal = normal_projection @ delta_T_vect_frac
        delta_T_normal_normal = np.dot(normal, delta_T_normal)
        delta_T_normal_norm = np.linalg.norm(delta_T_normal.T, ord=2, axis=1)
        delta_T_tangential = tangential_projetion @ delta_T_vect_frac
        delta_T_tangential_y = delta_T_tangential[
            1
        ].T  # one on-plane direction is aligned with y
        delta_T_tangential_t = np.linalg.norm(
            np.array([delta_T_tangential[0], delta_T_tangential[2]]).T, ord=2, axis=1
        )  # the other tangential dir, t, is made of x and z
        delta_T_tangential_norm = np.linalg.norm(delta_T_tangential.T, ord=2, axis=1)

        delta_cff = (
            delta_T_tangential_norm
            + T_tangential_norm_ref
            - friction_coeff * (delta_T_normal_normal + T_normal_normal_ref)
        ) - (
            T_tangential_norm_ref - friction_coeff * T_normal_normal_ref
        )  # this is in Pascal

        # NN: ---------------------------------------------------------------------------
        delta_T_nn = np.load(
            "./results/nn/"
            + str(idx_mu)
            + "/traction_fracture_vector_"
            + str(time)
            + ".npy"
        )

        delta_T_vect_frac_nn = np.reshape(delta_T_nn, (dim, -1), order="F")

        delta_T_normal_nn = normal_projection @ delta_T_vect_frac_nn
        delta_T_normal_normal_nn = np.dot(normal, delta_T_normal_nn)
        delta_T_normal_norm_nn = np.linalg.norm(delta_T_normal_nn.T, ord=2, axis=1)
        delta_T_tangential_nn = tangential_projetion @ delta_T_vect_frac_nn
        delta_T_tangential_y_nn = delta_T_tangential_nn[
            1
        ].T  # one on-plane direction is aligned with y
        delta_T_tangential_t_nn = np.linalg.norm(
            np.array([delta_T_tangential_nn[0], delta_T_tangential_nn[2]]).T,
            ord=2,
            axis=1,
        )  # the other tangential dir, t, is made of x and z
        delta_T_tangential_norm_nn = np.linalg.norm(
            delta_T_tangential_nn.T, ord=2, axis=1
        )

        delta_cff_nn = (
            delta_T_tangential_norm_nn
            + T_tangential_norm_ref
            - friction_coeff * (delta_T_normal_normal_nn + T_normal_normal_ref)
        ) - (
            T_tangential_norm_ref - friction_coeff * T_normal_normal_ref
        )  # this is in Pascal

        np.save(save_folder + "/cff_" + str(time), delta_cff_nn)

        # Deltas NN-FOM: ------------------------------------------------------------------------------
        delta_delta_T_vect_frac = delta_T_vect_frac_nn - delta_T_vect_frac
        delta_delta_T_normal_norm = delta_T_normal_norm_nn - delta_T_normal_norm
        delta_delta_T_normal_normal = delta_T_normal_normal_nn - delta_T_normal_normal
        delta_delta_T_tangential_norm = (
            delta_T_tangential_norm_nn - delta_T_tangential_norm
        )
        delta_delta_T_tangential_y = delta_T_tangential_y_nn - delta_T_tangential_y
        delta_delta_T_tangential_t = delta_T_tangential_t_nn - delta_T_tangential_t
        delta_delta_cff = delta_cff_nn - delta_cff

        delta_traction_tangential_norm_nn = delta_T_tangential_norm_nn * (
            face_area[fracture_faces_id]
        )  # in newton
        delta_traction_normal_normal_nn = delta_T_normal_normal_nn * (
            face_area[fracture_faces_id]
        )
        # delta_delta_traction_tangential_y = delta_delta_T_tangential_y * ( face_area[fracture_faces_id] )
        # delta_delta_traction_tangential_t = delta_delta_T_tangential_t * ( face_area[fracture_faces_id] )

        print("there is a weird change of sign due to face_area")
        pdb.set_trace()
        exporter.write_vtu(
            [
                (sd_fract, "T_xyz_nn", delta_T_vect_frac_nn),
                (sd_fract, "T_out_of_plane_nn", delta_T_normal_norm_nn),
                (sd_fract, "T_out_of_plane_signed_nn", delta_T_normal_normal_nn),
                (sd_fract, "minus_T_out_of_plane_signed_nn", -delta_T_normal_normal_nn),
                (sd_fract, "T_tangential_norm_nn", delta_T_tangential_norm_nn),
                (sd_fract, "T_tangential_y_nn", delta_T_tangential_y_nn),
                (sd_fract, "T_tangential_t_nn", delta_T_tangential_t_nn),
                (sd_fract, "cff_nn", delta_cff_nn),
                #
                (sd_fract, "delta_FOM_NN_T_xyz", delta_T_vect_frac),
                (sd_fract, "delta_FOM_NN_T_out_of_plane", delta_delta_T_normal_norm),
                (
                    sd_fract,
                    "delta_FOM_NN_T_out_of_plane_signed",
                    delta_delta_T_normal_normal,
                ),
                (
                    sd_fract,
                    "delta_FOM_NN_minus_T_out_of_plane_signed",
                    -delta_delta_T_normal_normal,
                ),
                (
                    sd_fract,
                    "delta_FOM_NN_T_tangential_norm",
                    delta_delta_T_tangential_norm,
                ),
                (sd_fract, "delta_FOM_NN_T_tangential_y", delta_delta_T_tangential_y),
                (sd_fract, "delta_FOM_NN_T_tangential_t", delta_delta_T_tangential_t),
                (sd_fract, "delta_FOM_NN_cff", delta_delta_cff),
                (
                    sd_fract,
                    "delta_traction_tangential_norm_nn",
                    delta_traction_tangential_norm_nn,
                ),  # added
                (
                    sd_fract,
                    "delta_traction_normal_normal_nn",
                    delta_traction_normal_normal_nn,
                ),  # added
                # (
                #     sd_fract,
                #     "delta_delta_traction_tangential_y",
                #     delta_delta_traction_tangential_y,
                # ),  # added # no, I dont need them
                # (
                #     sd_fract,
                #     "delta_delta_traction_tangential_t",
                #     delta_delta_traction_tangential_t,
                # ),  # added
            ],
            time_dependent=True,
            time_step=time,
        )

    arguments = np.array_split(times_mech, 3)
    with ThreadPoolExecutor() as executor:
        output_executor = executor.map(parallel_fcn, arguments)

    for i in output_executor:
        print(i)

    print("Traction from ROM computed!\n")


# -----------------------------------------------------------------------------------------------------
for idx_mu in test_dataset_id:
    print("postprocessing " + str(idx_mu))

    # # old:
    # pu.compute_fault_traction_fom(idx_mu)
    # pu.compute_fault_traction_nn_cut(idx_mu)

    compute_fault_traction_fom(idx_mu)
    compute_fault_traction_nn_cut(idx_mu)


print("\nDone!")
