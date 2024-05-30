import os
import sys
import pdb
import warnings
import inspect
import copy
import shutil


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
    # sys.path.append("/g100_work/pMI24_MatBa/eballin1/mypythonmodules")  # Cineca G100
import ppromode

sys.path.append("../porepy/src")  # lets add pp in mypythonmodules in G100
import porepy as pp

import sub_model_fom_case_eni

"""
"""


class ModelCaseEni:
    def __init__(self, data_folder):
        """ """
        self.data_folder = data_folder

        self.write_common_data()

    def write_common_data(self):
        """ """
        np.savetxt(self.data_folder + "/mech/NU", np.array([0.25]))

        t_0 = 1
        time_production = 20 * 365.25 / t_0  # 20 years, leap year included
        time_injection = 20 * 365.25 / t_0
        time_final_training = (
            time_production + time_injection
        )  # equal to the fluid and mech simulaition time
        timestep = time_final_training / 40 / t_0
        timestep_nn = 1 * timestep
        time_final_test = time_final_training

        np.savetxt(self.data_folder + "/TIME_PRODUCTION", np.array([time_production]))
        np.savetxt(self.data_folder + "/TIME_INJECTION", np.array([time_injection]))
        np.savetxt(self.data_folder + "/TIMESTEP", np.array([timestep]))
        np.savetxt(self.data_folder + "/TIMESTEP_NN", np.array([timestep_nn]))
        np.savetxt(
            self.data_folder + "/TRAINING_TIMES",
            np.arange(0, time_final_training + timestep_nn, timestep_nn),
        )
        np.savetxt(
            self.data_folder + "/TEST_TIMES",
            np.arange(0, time_final_test + timestep_nn, timestep_nn),
        )

    def run_one_simulation(
        self,
        data_folder_root: str,
        save_folder_root: str,
        idx_mu: int,
        mu_param: np.array,
    ) -> None:
        """
        - TODO: data folder and save folder are concept not totally clear, improve
        """
        times = np.loadtxt(
            data_folder_root + "/TIMES"
        )  # in this case, I dont have timestep chops, so I save only one TIMES file

        # mechanics:
        save_folder = save_folder_root + "/mech/" + str(idx_mu)

        pp_model = sub_model_fom_case_eni.SubModelCaseEni()
        pp_model.save_folder = save_folder
        pp_model.exporter_folder = save_folder
        pp_model.nu = np.loadtxt(data_folder_root + "/mech/NU")

        for time in times:
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
            pp_model.echelon_pressure = echelon_pressure
            pp_params = {}
            pp.run_stationary_model(pp_model, pp_params)

        del pp_model, pp_params

    def run_one_simulation_no_python(
        self,
        data_folder_root: str,
        save_folder_root: str,
        idx_mu,
        mu_param: np.array,
    ) -> None:
        """
        - TODO: data folder and save folder are concept not totally clear, improve
        - TODO: find a better way to write mu_param in the input file
        """

        # pore compressibility, c_pp
        E_ave = (mu_param[3] + mu_param[4]) / 2  # bad approximation
        with open(data_folder_root + "/fluid/PORO.INC") as f:
            lines = f.read().splitlines()[
                1:-2
            ]  # HARDCODED: first an last two elements are strings
            phi = np.array(lines, dtype=np.float32)

        phi_ave = (
            np.sum(phi) / phi.shape[0]
        )  # HARDOCED, and not consistent with previous choice...
        nu = np.loadtxt(data_folder_root + "/mech/NU")
        c_pp = (1 + nu) * (1 - 2 * nu) / ((1 - nu) * E_ave * phi)

        # modify .DATA and prepare running folder
        save_folder = save_folder_root + "/fluid/" + str(idx_mu)
        try:
            os.mkdir(save_folder)
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
        replacement_pattern = str(np.loadtxt(data_folder_root + "/TIME_PRODUCTION"))
        ppromode.replace_pattern(file_name, search_pattern, replacement_pattern)

        search_pattern = "__TIME_INJECTION__"
        replacement_pattern = str(np.loadtxt(data_folder_root + "/TIME_INJECTION"))
        ppromode.replace_pattern(file_name, search_pattern, replacement_pattern)

        search_pattern = "__PRODUCTION_RATE__"
        replacement_pattern = str(mu_param[6])
        ppromode.replace_pattern(file_name, search_pattern, replacement_pattern)

        search_pattern = "__INJECTION_RATE__"
        replacement_pattern = str(mu_param[5])
        ppromode.replace_pattern(file_name, search_pattern, replacement_pattern)

        search_pattern = "__TRANSMISS_PERP__"
        replacement_pattern = str(mu_param[1])
        ppromode.replace_pattern(file_name, search_pattern, replacement_pattern)

        # search_pattern = ""
        # replacement_pattern = ""
        # ppromode.replace_pattern(file_name, search_pattern, replacement_pattern)

        # search_pattern = ""
        # replacement_pattern = ""
        # ppromode.replace_pattern(file_name, search_pattern, replacement_pattern)

        # prepare run working folder
        shutil.copy(
            save_folder_root + "/fluid/" + "FAULTS.INC",
            save_folder_root + "/fluid/" + str(idx_mu) + "/" + "FAULTS"
            # + str(idx_mu)
            + ".INC",
        )

        shutil.copy(
            save_folder_root + "/fluid/" + "FIPNUM.INC",
            save_folder_root + "/fluid/" + str(idx_mu) + "/" + "FIPNUM"
            # + str(idx_mu)
            + ".INC",
        )
        shutil.copy(
            save_folder_root + "/fluid/" + "FLUXNUM.INC",
            save_folder_root + "/fluid/" + str(idx_mu) + "/" + "FLUXNUM"
            # + str(idx_mu)
            + ".INC",
        )
        shutil.copy(
            save_folder_root + "/fluid/" + "PERMX.INC",
            save_folder_root + "/fluid/" + str(idx_mu) + "/" + "PERMX"
            # + str(idx_mu)
            + ".INC",
        )
        shutil.copy(
            save_folder_root + "/fluid/" + "PERMY.INC",
            save_folder_root + "/fluid/" + str(idx_mu) + "/" + "PERMY"
            # + str(idx_mu)
            + ".INC",
        )
        shutil.copy(
            save_folder_root + "/fluid/" + "PERMZ.INC",
            save_folder_root + "/fluid/" + str(idx_mu) + "/" + "PERMZ"
            # + str(idx_mu)
            + ".INC",
        )
        shutil.copy(
            save_folder_root + "/fluid/" + "PORO.INC",
            save_folder_root + "/fluid/" + str(idx_mu) + "/" + "PORO"
            # + str(idx_mu)
            + ".INC",
        )

        shutil.copy(
            save_folder_root + "/fluid/" + "PUNQS3_EOS_COMPS.INC",
            save_folder_root + "/fluid/" + str(idx_mu) + "/" + "PUNQS3_EOS_COMPS"
            # + str(idx_mu)
            + ".INC",
        )

        shutil.copy(
            save_folder_root + "/fluid/" + "ROCK.INC",
            save_folder_root + "/fluid/" + str(idx_mu) + "/" + "ROCK"
            # + str(idx_mu)
            + ".INC",
        )

        shutil.copy(
            save_folder_root + "/fluid/" + "ROCKNUM.INC",
            save_folder_root + "/fluid/" + str(idx_mu) + "/" + "ROCKNUM"
            # + str(idx_mu)
            + ".INC",
        )

        shutil.copy(
            save_folder_root + "/fluid/" + "SCAL.INC",
            save_folder_root + "/fluid/" + str(idx_mu) + "/" + "SCAL"
            # + str(idx_mu)
            + ".INC",
        )

        shutil.copy(
            save_folder_root + "/fluid/" + "case2skew.EGRID",
            save_folder_root + "/fluid/" + str(idx_mu) + "/" + "case2skew"
            # + str(idx_mu)
            + ".EGRID",
        )

        os.system("cd " + data_folder_root + "/" + str(idx_mu))
        os.system("runPBSechelon case2skew.DATA")
