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
    
    times = np.loadtxt("./data/TIMES")
    
    exporter = pp.Exporter(
        sd, folder_name="./results/" + str(idx_mu), file_name="delta_displacement"
    )
    
    for time in times:
        u_ref = np.load("./data/mech/ref/displacement.npy")
        u = np.load("./data/mech/" + str(idx_mu) + "/displacement_" + str(time)+ ".npy")
    
        delta_u = u - u_ref
    
        exporter.write_vtu([(sd, "delta_displacement", delta_u)], time_step=time)


def compute_fault_traction(idx_mu):
    """ """
    save_folder = "./results/" + str(idx_mu)
    
    with open("./data/mech/" + str(idx_mu) + "/sd_fract.pkl", "rb") as fle:
        sd_fract = pickle.load(fle)
        
    exporter = pp.Exporter(
        sd_fract,
        file_name="sd_fract",
        folder_name=save_folder,
    )
    
    u_ref = np.load("./data/mech/ref/displacement.npy")
    times = np.loadtxt("./data/TIMES")
    
    for time in times:
        stress_tensor_grad = sp.sparse.load_npz(
            "./data/mech/" + str(idx_mu) + "/stress_tensor_grad_"+str(time)+".npz"
        )
        bound_stress = sp.sparse.load_npz(
            "./data/mech/" + str(idx_mu) + "/bound_stress_"+str(time)+".npz"
        )
    
        u = np.load("./data/mech/" + str(idx_mu) + "/displacement_"+str(time)+".npy")
    
        u_b_filled_ref = np.load("./data/mech/ref/displacement_boundary_"+str(time)+".npy")
        u_b_filled = np.load("./data/mech/" + str(idx_mu) + "/displacement_boundary_"+str(time)+".npy")
    
        dim = 3
        fracture_faces_id = np.load("./data/mech/" + str(idx_mu) + "/fracture_faces_id.npy")
        normal = np.load("./data/mech/" + str(idx_mu) + "/normal_"+str(time)+".npy")
    
        delta_u = u - u_ref
        delta_u_b_filled = u_b_filled - u_b_filled_ref
    
        # which one? ###
        T = stress_tensor_grad * u + bound_stress * u_b_filled
        T_vect = np.reshape(T, (dim, -1), order="F")
        T_vect_frac = T_vect[:, fracture_faces_id]
    
        # which one? ###
        delta_T = stress_tensor_grad * delta_u + bound_stress * delta_u_b_filled
        delta_T_vect = np.reshape(delta_T, (dim, -1), order="F")
        delta_T_vect_frac = delta_T_vect[:, fracture_faces_id]
    
        T = delta_T  ###
        T_vect = delta_T_vect  ###
        T_vect_frac = delta_T_vect_frac  ###
    
        # pp.plot_grid(sd, vector_value=T_vect, figsize=(15, 12), alpha=0)
        # pp.plot_grid(
        #     pp_model.sd_fract, T_vect_frac, alpha=0
        # )  # NO, for pp pp_model.sd_fract is 2D, T_vect_frac is 3D, so they don't match, see below
        # T_vect_frac_filled = np.zeros((pp_model.nd, sd.num_faces))
        # T_vect_frac_filled[:, pp_model.fracture_faces_id] = T_vect_frac
        # pp.plot_grid(sd, vector_value=10000 * T_vect_frac_filled, alpha=0) # there is an eror in paraview... don't trust it
    
        
        exporter.write_vtu(
            [
                (sd_fract, "T_x", T_vect_frac[0]),
                (sd_fract, "T_y", T_vect_frac[1]),
                (sd_fract, "T_z", T_vect_frac[2]),
            ], time_step = time,
        )
    
        normal = normal / np.linalg.norm(normal, ord=2)
        normal_projection = pp.map_geometry.normal_matrix(normal=normal)
        tangential_projetion = pp.map_geometry.tangent_matrix(normal=normal)
    
        T_normal = normal_projection @ T_vect_frac
        T_normal_norm = np.linalg.norm(T_normal.T, ord=2, axis=1)
        T_tangential = tangential_projetion @ T_vect_frac
        T_tangential_norm = np.linalg.norm(T_tangential.T, ord=2, axis=1)

    print("\nTraction computed!")


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
