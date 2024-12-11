import os
import sys
import pdb
import numpy as np

pp_path = "../porepy/src"
if pp_path not in sys.path:
    sys.path.append(pp_path)

import porepy as pp

os.system("clear")

data_folder = "/media/inspiron/DATI Toshiba/here/data"
results_folder_mech = "./results/mech"
results_folder_nn = "./results/nn"

training_dataset_id = np.loadtxt(data_folder + "/training_dataset_id", dtype=np.int32)
validation_dataset_id = np.loadtxt(
    data_folder + "/validation_dataset_id", dtype=np.int32
)
test_dataset_id = np.loadtxt(data_folder + "/test_dataset_id", dtype=np.int32)

times_mech = np.loadtxt(data_folder + "/TIMES_MECH")
dim = 3

friction_coeff = 0.45  # form spe paper


for idx_mu in test_dataset_id:
    print("computing cff of " + str(idx_mu))

    err_rel = 999999999 * np.ones(times_mech.shape[0])
    err_abs = 999999999 * np.ones(times_mech.shape[0])
    cff_fom_integrated_list = 999999999 * np.ones(times_mech.shape[0])
    cff_nn_integrated_list = 999999999 * np.ones(times_mech.shape[0])
    cff_fom_list = 999999999 * np.ones(times_mech.shape[0])
    cff_nn_list = 999999999 * np.ones(times_mech.shape[0])

    normal_area = np.load(data_folder + "/mech/800/normal_area" + ".npy")
    normal = normal_area / np.linalg.norm(normal_area, ord=2)
    normal_projection = pp.map_geometry.normal_matrix(normal=normal)
    tangential_projetion = pp.map_geometry.tangent_matrix(normal=normal)
    vols = np.load(data_folder + "/mech/800/volumes_subdomains.npy")
    vols = vols[range(int(vols.shape[0] / dim))]

    for idx_dt, time in enumerate(times_mech):
        # fom: ------------------
        # T_vect_frac = np.reshape( np.load(data_folder + "/mech/" + str(idx_mu) + "/traction_fracture_vector_" + str(time) + ".npy"), (dim, -1), order="F")

        # T_normal = normal_projection @ T_vect_frac
        # T_normal_normal = np.dot(normal, T_normal)
        # T_normal_norm = np.linalg.norm(T_normal.T, ord=2, axis=1)
        # T_tangential = tangential_projetion @ T_vect_frac
        # T_tangential_y = T_tangential[1].T # one on-plane direction is aligned with y
        # T_tangential_t = np.linalg.norm(np.array([T_tangential[0],T_tangential[2]]).T, ord=2, axis=1) # the other tangential dir, t, is made of x and z
        # T_tangential_norm = np.linalg.norm(T_tangential.T, ord=2, axis=1)

        # cff_fom = T_tangential_norm - friction_coeff * T_normal_normal

        # np.save(results_folder_mech + "/" + str(idx_mu) + "/cff_" + str(time), cff_fom)

        cff_fom = np.load(
            results_folder_mech + "/" + str(idx_mu) + "/cff_" + str(time) + ".npy"
        )

        # nn: -------------------------
        # T_vect_frac = np.reshape( np.load(results_folder_nn + "/" + str(idx_mu) + "/traction_fracture_vector_" + str(time) + ".npy"), (dim, -1), order="F")

        # T_normal = normal_projection @ T_vect_frac
        # T_normal_normal = np.dot(normal, T_normal)
        # T_normal_norm = np.linalg.norm(T_normal.T, ord=2, axis=1)
        # T_tangential = tangential_projetion @ T_vect_frac
        # T_tangential_y = T_tangential[1].T # one on-plane direction is aligned with y
        # T_tangential_t = np.linalg.norm(np.array([T_tangential[0],T_tangential[2]]).T, ord=2, axis=1) # the other tangential dir, t, is made of x and z
        # T_tangential_norm = np.linalg.norm(T_tangential.T, ord=2, axis=1)

        # cff_nn = T_tangential_norm - friction_coeff * T_normal_normal

        # np.save(results_folder_nn + "/" + str(idx_mu) + "/cff_" + str(time), cff_nn)

        cff_nn = np.load(
            results_folder_nn + "/" + str(idx_mu) + "/cff_" + str(time) + ".npy"
        )

        # err: ----------------
        ref = max(np.dot(cff_fom**2, vols), 1e-12)
        diff = cff_fom - cff_nn  # in Pa
        err_rel[idx_dt] = np.sqrt(np.dot(diff**2, vols) / ref)  # dimless
        # err_abs[idx_dt] = np.sqrt(
        #                 np.dot(diff**2, vols)
        #             ) # in N
        err_abs[idx_dt] = np.sqrt(np.dot(diff**2, vols) / sum(vols))  # in Pa

        cff_fom_integrated_list[idx_dt] = np.sqrt(np.dot(cff_fom**2, vols))
        # cff_nn_integrated_list[idx_dt] = np.sqrt(np.dot(cff_nn**2, vols))

        # cff_fom_list[idx_dt] = np.sqrt(np.dot(cff_fom**2, vols) / sum(vols) )
        # cff_nn_list[idx_dt] = np.sqrt(np.dot(cff_nn**2, vols) / sum(vols))

        np.save(
            results_folder_mech + "/" + str(idx_mu) + "/cff_fom_" + str(time), cff_fom
        )
        np.save(results_folder_nn + "/" + str(idx_mu) + "/cff_nn_" + str(time), cff_nn)

    np.savetxt(results_folder_nn + "/" + str(idx_mu) + "/err_relative_cff", err_rel)
    np.savetxt(results_folder_nn + "/" + str(idx_mu) + "/err_area_cff", err_abs)
    np.savetxt(
        results_folder_mech + "/" + str(idx_mu) + "/cff_fom_integrated",
        cff_fom_integrated_list,
    )
    # np.savetxt(results_folder_nn + "/" + str(idx_mu) + "/cff_nn_integrated_space_ave", cff_nn_integrated_list)
    # np.savetxt(results_folder_mech + "/" + str(idx_mu) + "/cff_fom_space_ave", cff_fom_list) ###
    # np.savetxt(results_folder_nn + "/" + str(idx_mu) + "/cff_nn_space_ave", cff_nn_list) ###


err_rel_cff_mu_time = 999999999 * np.ones(
    (test_dataset_id.shape[0], times_mech.shape[0])
)
err_abs_cff_mu_time = 999999999 * np.ones(
    (test_dataset_id.shape[0], times_mech.shape[0])
)
# cff_fom_mu_time = 999*np.ones((test_dataset_id.shape[0], times_mech.shape[0]))
# cff_nn_mu_time = 999*np.ones((test_dataset_id.shape[0], times_mech.shape[0]))
cff_fom_integrated = 999999999 * np.ones(
    (test_dataset_id.shape[0], times_mech.shape[0])
)
err_rel_vs_max_cff_mu_time = 999999999 * np.ones(
    (test_dataset_id.shape[0], times_mech.shape[0])
)

for i, idx_mu in enumerate(test_dataset_id):
    err_rel_cff_mu_time[i] = np.loadtxt(
        results_folder_nn + "/" + str(idx_mu) + "/err_relative_cff"
    )
    err_abs_cff_mu_time[i] = np.loadtxt(
        results_folder_nn + "/" + str(idx_mu) + "/err_area_cff"
    )
    # cff_fom_mu_time[i] = np.loadtxt(results_folder_mech + "/" + str(idx_mu) + "/cff_fom_space_ave") ###
    # cff_nn_mu_time[i] = np.loadtxt(results_folder_nn + "/" + str(idx_mu) + "/cff_nn_space_ave") ###
    cff_fom_integrated[i] = np.loadtxt(
        results_folder_mech + "/" + str(idx_mu) + "/cff_fom_integrated"
    )


# np.savetxt(results_folder_mech + "/cff_fom", cff_fom_mu_time)
# np.savetxt(results_folder_nn + "/cff_nn", cff_nn_mu_time)
np.savetxt(results_folder_mech + "/cff_fom_integrated", cff_fom_integrated)

err_ave_mu_rel = np.sum(err_rel_cff_mu_time, axis=0) / err_rel_cff_mu_time.shape[0]
err_ave_mu_abs = np.sum(err_abs_cff_mu_time, axis=0) / err_abs_cff_mu_time.shape[0]
np.savetxt(results_folder_nn + "/err_relative_cff_ave", err_ave_mu_rel)
np.savetxt(results_folder_nn + "/err_area_cff_ave", err_ave_mu_abs)


# compute err_rel_vs_max
cff_fom_integrated = np.loadtxt(results_folder_mech + "/cff_fom_integrated")
cff_fom_integrated_max = np.max(cff_fom_integrated, axis=0)
cff_nn_list = 999999999 * np.ones(times_mech.shape[0])
cff_max_list = 999999999 * np.ones((test_dataset_id.shape[0], times_mech.shape[0]))


for i, idx_mu in enumerate(test_dataset_id):
    print("computing cff of " + str(idx_mu))
    err_rel = 999999999 * np.ones(times_mech.shape[0])
    vols = np.load(data_folder + "/mech/800/volumes_subdomains.npy")
    vols = vols[range(int(vols.shape[0] / dim))]

    for idx_dt, time in enumerate(times_mech):
        cff_fom = np.load(
            results_folder_mech + "/" + str(idx_mu) + "/cff_fom_" + str(time) + ".npy"
        )
        cff_nn = np.load(
            results_folder_nn + "/" + str(idx_mu) + "/cff_nn_" + str(time) + ".npy"
        )
        ref = max(
            cff_fom_integrated_max[idx_dt], 1e-12
        )  # time 0 is included in this loop...
        diff = cff_fom - cff_nn
        err_rel[idx_dt] = np.sqrt(np.dot(diff**2, vols)) / ref  # ref is already a sqrt

        cff_max_list[i, idx_dt] = np.max(cff_fom)

    np.savetxt(
        results_folder_nn + "/" + str(idx_mu) + "/err_relative_vs_max_cff", err_rel
    )

    err_rel_vs_max_cff_mu_time[i] = np.loadtxt(
        results_folder_nn + "/" + str(idx_mu) + "/err_relative_vs_max_cff"
    )

# sometimes the following lines dont work, ?? DONT OPEN THE FILE WITH NOTEPAD!
idx_mu_max_err_rel = np.argmax(err_rel_cff_mu_time, axis=0) + test_dataset_id[0]
np.savetxt(results_folder_nn + "/idx_mu_max_err_rel_cff", idx_mu_max_err_rel)

idx_mu_max_err_abs = np.argmax(err_abs_cff_mu_time, axis=0) + test_dataset_id[0]
np.savetxt(results_folder_nn + "/idx_mu_max_err_area_cff", idx_mu_max_err_abs)

idx_mu_max_err_rel_vs_max_cff = (
    np.argmax(err_rel_vs_max_cff_mu_time, axis=0) + test_dataset_id[0]
)
np.savetxt(
    results_folder_nn + "/idx_mu_max_err_rel_vs_max_cff", idx_mu_max_err_rel_vs_max_cff
)
np.savetxt(
    results_folder_nn + "/err_rel_vs_max_cff_mu_time", err_rel_vs_max_cff_mu_time
)

idx_max_cff = np.argmax(cff_max_list, axis=0) + test_dataset_id[0]
np.savetxt(results_folder_nn + "/idx_max_cff", idx_max_cff)

print("\nDone!")
