import sys
import os
import pdb
import pickle
import numpy as np
import scipy as sp

pp_path = "../porepy/src"
if pp_path not in sys.path:
    sys.path.append(pp_path)

import porepy as pp


os.system("clear")


def compute_and_plot_delta_displacement(idx_mu):
    """ """
    import sub_model_fom_case_eni

    geometry = sub_model_fom_case_eni.GeometryCloseToEni()
    geometry.set_geometry()
    sd = geometry.mdg.subdomains(dim=3)[0]

    # times = np.loadtxt("./data/TIMES")
    times_mech = np.loadtxt("./data/TIMES_MECH")

    exporter = pp.Exporter(
        sd, folder_name="./results/mech/" + str(idx_mu), file_name="u_and_p"
    )

    for time in times_mech:
        print("time = ", time)
        # u_ref = np.load("./data/mech/ref/displacement.npy")
        u_ref =  u = np.load(
            "./data/mech/" + str(idx_mu) + "/displacement_" + str(0.0) + ".npy"
        )
        u = np.load(
            "./data/mech/" + str(idx_mu) + "/displacement_" + str(time) + ".npy"
        )

        delta_u = u - u_ref
        
        p = np.load(
            "./data/fluid/" + str(idx_mu) + "/fluid_pressure_" + str(time) + ".npy"
        )

        exporter.write_vtu(
            [(sd, "delta_displacement", delta_u),
             (sd, "fluid_pressure", p)], 
            time_dependent=True, time_step=time
        )
    
    print("delta displacement computed!\n")


def compute_fault_traction(idx_mu):
    """ """
    save_folder = "./results/mech/" + str(idx_mu)

    with open("./data/mech/" + str(idx_mu) + "/sd_fract.pkl", "rb") as fle:
        sd_fract = pickle.load(fle)

    exporter = pp.Exporter(
        sd_fract,
        file_name="sd_fract",
        folder_name=save_folder,
    )

    # u_ref = np.load("./data/mech/ref/displacement.npy")
    u_ref =  u = np.load(
        "./data/mech/" + str(idx_mu) + "/displacement_" + str(0.0) + ".npy"
    )
    # u_b_filled_ref = np.load("./data/mech/ref/displacement_boundary.npy")
    u_b_filled_ref = np.load(
        "./data/mech/"
        + str(idx_mu)
        + "/displacement_boundary_"
        + str(0.0)
        + ".npy"
    )
    # times = np.loadtxt("./data/TIMES")
    times_mech = np.loadtxt("./data/TIMES_MECH")


    for time in times_mech:
        print("time = ", time)
        stress_tensor_grad = sp.sparse.load_npz(
            "./data/mech/" + str(idx_mu) + "/stress_tensor_grad_" + str(time) + ".npz"
        )
        bound_stress = sp.sparse.load_npz(
            "./data/mech/" + str(idx_mu) + "/bound_stress_" + str(time) + ".npz"
        )

        u = np.load(
            "./data/mech/" + str(idx_mu) + "/displacement_" + str(time) + ".npy"
        )

        u_b_filled = np.load(
            "./data/mech/"
            + str(idx_mu)
            + "/displacement_boundary_"
            + str(time)
            + ".npy"
        )

        dim = 3
        fracture_faces_id = np.load(
            "./data/mech/" + str(idx_mu) + "/fracture_faces_id.npy"
        )
        normal_area =  np.load("./data/mech/" + str(idx_mu) + "/normal_area" + ".npy")
        face_area = np.load("./data/mech/" + str(idx_mu) + "/face_area" + ".npy")
        #normal = normal / faces_area
        normal = normal_area / np.linalg.norm(normal_area, ord=2)

        delta_u = u - u_ref
        delta_u_b_filled = u_b_filled - u_b_filled_ref

        # # which one?
        # T = stress_tensor_grad * u + bound_stress * u_b_filled
        # T_vect = np.reshape(T, (dim, -1), order="F")
        # T_vect_frac = T_vect[:, fracture_faces_id]

        # which one? # this one
        delta_T_area = stress_tensor_grad * delta_u + bound_stress * delta_u_b_filled
        delta_T_vect = np.reshape(delta_T_area, (dim, -1), order="F") / face_area
        delta_T_vect_frac = delta_T_vect[:, fracture_faces_id]

        T_area = delta_T_area 
        T_vect = delta_T_vect  
        T_vect_frac = delta_T_vect_frac  

        # pp.plot_grid(sd, vector_value=T_vect, figsize=(15, 12), alpha=0)
        # pp.plot_grid(
        #     pp_model.sd_fract, T_vect_frac, alpha=0
        # )  # NO, for pp pp_model.sd_fract is 2D, T_vect_frac is 3D, so they don't match, see below
        # T_vect_frac_filled = np.zeros((pp_model.nd, sd.num_faces))
        # T_vect_frac_filled[:, pp_model.fracture_faces_id] = T_vect_frac
        # pp.plot_grid(sd, vector_value=10000 * T_vect_frac_filled, alpha=0) # there is an eror in paraview... don't trust it
       
        normal_projection = pp.map_geometry.normal_matrix(normal=normal)
        tangential_projetion = pp.map_geometry.tangent_matrix(normal=normal)

        T_normal = normal_projection @ T_vect_frac
        T_normal_norm = np.linalg.norm(T_normal.T, ord=2, axis=1)
        T_tangential = tangential_projetion @ T_vect_frac
        T_tangential_y = T_tangential[1].T # one on-plane direction is aligned with y
        T_tangential_t = np.linalg.norm(np.array([T_tangential[0],T_tangential[2]]).T, ord=2, axis=1) # the other tangential dir, t, is made of x and z
        T_tangential_norm = np.linalg.norm(T_tangential.T, ord=2, axis=1)
      
        # exporter.write_vtu(
        #      [
        #          (sd_fract, "T_x", T_vect_frac[0]),
        #          (sd_fract, "T_y", T_vect_frac[1]),
        #          (sd_fract, "T_z", T_vect_frac[2]),
        #          (sd_fract, "T_n", T_normal_norm),
        #          (sd_fract, "T_tangential_y", T_tangential_y),
        #          (sd_fract, "T_tangential_t", T_tangential_t),
        #      ],
        #      time_dependent=True,
        #      time_step=time,
        #  )
        
        exporter.write_vtu(
             [
                 (sd_fract, "T_xyz", T_vect_frac),
                 (sd_fract, "T_on_plane", T_normal_norm),
                 (sd_fract, "T_tangential_y", T_tangential_y),
                 (sd_fract, "T_tangential_t", T_tangential_t),
             ],
             time_dependent=True,
             time_step=time,
         )
         
        
    print("Traction computed!\n")



def run_all_postprocess():
    """ """
    # data_folder_root = "./data"
    # save_folder_root = "./data"
    # model_fom = model_fom_case_eni.ModelCaseEni(data_folder_root, save_folder_root)
    training_dataset_id = np.loadtxt(data_folder_root + "/training_dataset_id")
    validation_dataset_id = np.loadtxt(data_folder_root + "/validation_dataset_id")
    test_dataset_id = np.loadtxt(data_folder_root + "/test_dataset_id")
    
    all_id_mu = np.concatenate(
        (training_dataset_id, validation_dataset_id, test_dataset_id)
    )
     
    for idx_mu in idx_mu_list:
         compute_and_plot_delta_displacement(idx_mu)
         compute_fault_traction(idx_mu)


    # FIGURES
    
    # load rom_model
    
    # compute_and_plot_delta_displacement(idx_mu)
    # compute_fault_traction(idx_mu)


if __name__ == "__main__":

    import model_fom_case_eni

    data_folder = "./data"
    save_folder = "./data"
    model_fom = model_fom_case_eni.ModelCaseEni(data_folder, save_folder)
    # model_fom.run_ref_mechanics()

    idx_mu_list = np.array([99999])

    for idx_mu in idx_mu_list:
        compute_and_plot_delta_displacement(idx_mu)
        compute_fault_traction(idx_mu)
