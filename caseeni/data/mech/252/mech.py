import sys
import os
sys.path.append("../../../../porepy/src")
sys.path.append("../../../")

import numpy as np
import time as moduletime
import porepy as pp
import sub_model_fom_case_eni

data_folder_root = "../../"
save_folder_root = "../../"

times_mech = np.loadtxt(data_folder_root + "/TIMES_MECH")

# mechanics:
idx_mu = 252

mu_param = np.load(data_folder_root + "/mu_param_" + str(idx_mu) + ".npy")

save_folder = save_folder_root + "/mech/" + str(idx_mu)

if os.path.isfile(save_folder + "/end_file"):
    print("mech idx_mu " + str(idx_mu) + " already done")
else:
    for time in times_mech:
        # for time in [times[0]]:
        print("idx_mu = ", idx_mu, ", time = ", time)
        pp_model = sub_model_fom_case_eni.SubModelCaseEni()
        pp_model.save_folder = save_folder
        pp_model.exporter_folder = save_folder
        pp_model.mrst_folder = "../../"
        pp_model.nu = np.loadtxt(data_folder_root + "/mech/NU")

        pp_model.subscript = "_" + str(time)
        pp_model.mu_param = mu_param
        echelon_pressure = np.load(
            data_folder_root
            + "/fluid/"
            + str(idx_mu)
            + "/fluid_pressure_"
            + str(time)
            + ".npy"
        )
        pp_model.echelon_pressure = 1e5 * echelon_pressure  # bar in ech, Pa in pp
        # pp_model.echelon_pressure = None
        pp_params = {}
        t_1 = moduletime.time()
        pp.run_stationary_model(pp_model, pp_params)
        print(
            "one timestep of idx_mu = ", idx_mu, " took ", moduletime.time() - t_1
        )

    del pp_model, pp_params
    np.savetxt(save_folder + "/end_file")
        