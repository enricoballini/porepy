import os
import sys
import pdb
import warnings
import inspect
import copy
import shutil
import time as moduletime


from functools import partial
from typing import Callable, Optional, Sequence, cast


import numpy as np
import scipy as sp


my_modules_path = "../../mypythonmodules"
sentinel = False
for i in sys.path:
    if i == my_modules_path:
        sentinel = True
if not sentinel:
    sys.path.append(my_modules_path)
from nnrom.utilsode.misc_ode import replace_pattern

sys.path.append("../porepy/src")


"""
"""


class ModelCaseEni:
    def __init__(self, data_folder_root, save_folder_root):
        """
        - data_folder: where data are found, if any
        - save_folder: where to save offline data. For fluid data_folder=save_folder, for mech data_folder=fluid save folder, save folder=mech save folder
        """
        self.data_folder_root = data_folder_root
        self.save_folder_root = save_folder_root

        self.write_common_data()

    def write_common_data(self):
        """ """
        np.savetxt(self.data_folder_root + "/mech/NU", np.array([0.25]))

        t_0 = 1
        time_production = 20 * 365.25 / t_0  # 20 years, leap year included
        time_injection = 20 * 365.25 / t_0
        time_final_training = (
            time_production + time_injection
        )  # equal to the fluid and mech simulaition time
        timestep = time_final_training / 40 / t_0
        timestep_nn = 1 * timestep
        time_final_test = time_final_training

        np.savetxt(
            self.data_folder_root + "/TIME_PRODUCTION", np.array([time_production])
        )
        np.savetxt(
            self.data_folder_root + "/TIME_INJECTION", np.array([time_injection])
        )
        np.savetxt(self.data_folder_root + "/TIMESTEP", np.array([timestep]))
        np.savetxt(self.data_folder_root + "/TIMESTEP_NN", np.array([timestep_nn]))
        np.savetxt(
            self.data_folder_root + "/TRAINING_TIMES",
            np.arange(0, time_final_training + timestep_nn, timestep_nn),
        )
        np.savetxt(
            self.data_folder_root + "/TIMES",
            np.arange(0, time_final_training + timestep_nn, timestep_nn),
        )
        np.savetxt(
            self.data_folder_root + "/TEST_TIMES",
            np.arange(0, time_final_test + timestep_nn, timestep_nn),
        )
        np.savetxt(self.data_folder_root + "/TIMES_MECH", np.array([0, 20*365.25, 40*365.25]))
        
    def run_ref_fluid(self):
        """ """
        mu_param = np.array([np.log(1e0), np.log(1e0), 1, 5.71e10, 1.0, 0.0, 0.0])
        self.run_one_simulation_no_python("ref", mu_param)
        
        
    def run_ref_mechanics(self):
        """reference solution or initial state"""
        import porepy as pp
        import sub_model_fom_case_eni

        model = sub_model_fom_case_eni.SubModelCaseEni()
        model.mu_param = np.array([None, None, 1.0, 5.71e10])  # E ratio, E reservoir
        model.echelon_pressure = "gravity_only"
        os.system("mkdir ./data/mech/ref")
        model.save_folder = "./data/mech/ref"
        model.exporter_folder = "./data/mech/ref"
        model.subscript = ""
        model.nu = 0.25

        pp.run_stationary_model(model, {})
    

    def run_one_simulation(
        self,
        # data_folder_root: str,
        # save_folder_root: str,
        idx_mu: int,
        mu_param: np.array,
    ) -> None:
        """ """

        import porepy as pp
        import sub_model_fom_case_eni

        data_folder_root = self.data_folder_root
        save_folder_root = self.save_folder_root

        # times = np.loadtxt(
        #     data_folder_root + "/TIMES"
        # )  # in this case, I dont have timestep chops, so I save only one TIMES file
        
        times_mech = np.loadtxt(
            data_folder_root + "/TIMES_MECH"
        )  


        # mechanics:
        save_folder = save_folder_root + "/mech/" + str(idx_mu)
        # try:
        #     os.mkdir(data_folder)
        # except:
        #     pass
        
        for time in times_mech:
        # for time in [times[0]]:
            print("idx_mu = ", idx_mu, ", time = ", time)
            pp_model = sub_model_fom_case_eni.SubModelCaseEni()
            pp_model.save_folder = save_folder
            pp_model.exporter_folder = save_folder
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
            pp_model.echelon_pressure = 1e5 * echelon_pressure # bar in ech, Pa in pp
            # pp_model.echelon_pressure = None
            pp_params = {}
            t_1 = moduletime.time()
            pp.run_stationary_model(pp_model, pp_params)
            print("one timestep of idx_mu = ", idx_mu, " took ", moduletime.time()-t_1 )

        del pp_model, pp_params

    def run_one_simulation_no_python(
        self,
        # data_folder_root: str,
        # save_folder_root: str,
        idx_mu,
        mu_param: np.array,
    ) -> None:
        """
        - TODO: find a better way to write mu_param in the input file
        """
        data_folder_root = self.data_folder_root

        # pore compressibility, c_pp
        E_ave = mu_param[3]  # I dont need an ave since I simulate only the reservoir
        with open(data_folder_root + "/fluid/PORO.INC") as f:
            lines = f.read().splitlines()[
                1:-2
            ]  # HARDCODED: first an last two elements are strings
            phi = np.array(lines, dtype=np.float32)

        # phi_ave = np.sum(phi) / phi.shape[0]  # phi 2 = reservoir = 0.2 #
        phi_ave = np.max(phi)  # HARDCODED
        
        nu = np.loadtxt(data_folder_root + "/mech/NU")
        c_pp = (
            (1 + nu) * (1 - 2 * nu) / ((1 - nu) * (E_ave / 1e5) * phi_ave)
        )  # /1e5: we want echelon in bar

        # modify .DATA and prepare running folder
        # save_folder = save_folder_root + "/fluid/" + str(idx_mu)
        data_folder = data_folder_root + "/fluid/" + str(idx_mu)
        try:
            os.mkdir(data_folder)
        except:
            pass

        shutil.copy(
            data_folder_root + "/fluid/case2skew.DATA",
            data_folder_root + "/fluid/" + str(idx_mu) + "/case2skew"
            # + str(idx_mu)
            + ".DATA",
        )

        file_name = (
            data_folder_root
            + "/fluid/"
            + str(idx_mu)
            + "/case2skew"
            # + str(idx_mu)
            + ".DATA"
        )

        search_pattern = "__TIME_PRODUCTION__"
        tot_time = np.loadtxt(data_folder_root + "/TIME_PRODUCTION")
        replacement_pattern = str(int(tot_time / 365.25)) + "*365.25"
        replace_pattern(file_name, search_pattern, replacement_pattern)

        search_pattern = "__TIME_INJECTION__"
        tot_time = np.loadtxt(data_folder_root + "/TIME_INJECTION")
        replacement_pattern = str(int(tot_time / 365.25)) + "*365.25"
        replace_pattern(file_name, search_pattern, replacement_pattern)

        search_pattern = "__PRODUCTION_RATE__"
        replacement_pattern = str(mu_param[6])
        replace_pattern(file_name, search_pattern, replacement_pattern)

        search_pattern = "__INJECTION_RATE__"
        replacement_pattern = str(mu_param[5])
        replace_pattern(file_name, search_pattern, replacement_pattern)

        search_pattern = "__PERMEABILITY_X_MULTIPLIER__"
        replacement_pattern = str(np.exp(mu_param[0]))
        replace_pattern(file_name, search_pattern, replacement_pattern)

        search_pattern = "__TRANSMISS_PERP__"
        replacement_pattern = str(np.exp(mu_param[1]))
        replace_pattern(file_name, search_pattern, replacement_pattern)

        search_pattern = "__ROCK_COMPRESSIBILITY_1__"
        replacement_pattern = str(c_pp)
        replace_pattern(file_name, search_pattern, replacement_pattern)

        search_pattern = "__ROCK_COMPRESSIBILITY_2__"
        replacement_pattern = str(c_pp)
        replace_pattern(file_name, search_pattern, replacement_pattern)

        # prepare run working folder
        shutil.copy(
            data_folder_root + "/fluid/" + "FAULTS.INC",
            data_folder_root + "/fluid/" + str(idx_mu) + "/" + "FAULTS"
            # + str(idx_mu)
            + ".INC",
        )

        shutil.copy(
            data_folder_root + "/fluid/" + "FIPNUM.INC",
            data_folder_root + "/fluid/" + str(idx_mu) + "/" + "FIPNUM"
            # + str(idx_mu)
            + ".INC",
        )
        shutil.copy(
            data_folder_root + "/fluid/" + "FLUXNUM.INC",
            data_folder_root + "/fluid/" + str(idx_mu) + "/" + "FLUXNUM"
            # + str(idx_mu)
            + ".INC",
        )
        shutil.copy(
            data_folder_root + "/fluid/" + "PERMX.INC",
            data_folder_root + "/fluid/" + str(idx_mu) + "/" + "PERMX"
            # + str(idx_mu)
            + ".INC",
        )
        shutil.copy(
            data_folder_root + "/fluid/" + "PERMY.INC",
            data_folder_root + "/fluid/" + str(idx_mu) + "/" + "PERMY"
            # + str(idx_mu)
            + ".INC",
        )
        shutil.copy(
            data_folder_root + "/fluid/" + "PERMZ.INC",
            data_folder_root + "/fluid/" + str(idx_mu) + "/" + "PERMZ"
            # + str(idx_mu)
            + ".INC",
        )
        shutil.copy(
            data_folder_root + "/fluid/" + "PORO.INC",
            data_folder_root + "/fluid/" + str(idx_mu) + "/" + "PORO"
            # + str(idx_mu)
            + ".INC",
        )

        shutil.copy(
            data_folder_root + "/fluid/" + "PUNQS3_EOS_COMPS.INC",
            data_folder_root + "/fluid/" + str(idx_mu) + "/" + "PUNQS3_EOS_COMPS"
            # + str(idx_mu)
            + ".INC",
        )

        shutil.copy(
            data_folder_root + "/fluid/" + "ROCK.INC",
            data_folder_root + "/fluid/" + str(idx_mu) + "/" + "ROCK"
            # + str(idx_mu)
            + ".INC",
        )

        shutil.copy(
            data_folder_root + "/fluid/" + "ROCKNUM.INC",
            data_folder_root + "/fluid/" + str(idx_mu) + "/" + "ROCKNUM"
            # + str(idx_mu)
            + ".INC",
        )

        shutil.copy(
            data_folder_root + "/fluid/" + "SCAL.INC",
            data_folder_root + "/fluid/" + str(idx_mu) + "/" + "SCAL"
            # + str(idx_mu)
            + ".INC",
        )

        shutil.copy(
            data_folder_root + "/fluid/" + "case2skew.EGRID",
            data_folder_root + "/fluid/" + str(idx_mu) + "/" + "case2skew"
            # + str(idx_mu)
            + ".EGRID",
        )

        shutil.copy(
            data_folder_root + "/fluid/" + "GRID3D.GRDECL",
            data_folder_root + "/fluid/" + str(idx_mu) + "/" + "GRID3D"
            # + str(idx_mu)
            + ".GRDECL",
        )  # no idea, echelon looks for it

        os.system(
            "cd "
            + data_folder_root
            + "/fluid/"
            + str(idx_mu)
            + " && runPBSEchelon case2skew.DATA"
        )
        print(str(idx_mu) + " launced!")
