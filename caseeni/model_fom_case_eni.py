import os
import sys
import pdb
import warnings
import inspect
import copy
import shutil
import fileinput

from functools import partial
from typing import Callable, Optional, Sequence, cast


import numpy as np
import scipy as sp

if "/home/inspiron/Desktop/PhD/porepy/src" in sys.path:
    sys.path.remove("/home/inspiron/Desktop/PhD/porepy/src")
    sys.path.append("/home/inspiron/Desktop/PhD/eni_venv/porepy/src")
# sys.path.append("/g100_work/pMI24_MatBa/eballin1/eni_venv/porepy/src")

import porepy as pp

import sub_model_fom_case_eni

"""
"""


def replace_all(file_name, search_pattern, replace_patter):
    """ """
    for line in fileinput.input(file_name, inplace=1):
        if search_pattern in line:
            line = line.replace(search_pattern, replace_patter)
        sys.stdout.write(line)


class ModelCaseEni:
    def __init__(self, data_folder):
        """ """
        self.data_folder = data_folder

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

        # modifiy .DATA
        save_folder = save_folder_root + "/fluid/" + str(idx_mu)
        try:
            os.mkdir(save_folder)
        except:
            pass

        shutil.copy(
            data_folder_root + "/fluid/case2skew.DATA",
            data_folder_root
            + "/fluid/"
            + str(idx_mu)
            + "/case2skew_"
            + str(idx_mu)
            + ".DATA",
        )

        # with open(
        #     save_folder_root
        #     + "/fluid/"
        #     + str(idx_mu)
        #     + "/case2skew_"
        #     + str(idx_mu)
        #     + ".DATA",
        #     mode="w",
        # ) as f:
        #     datafile = f.readlines()
        #     pdb.set_trace()
        #     for line in datafile:
        #         print(line)
        #         if " W1 GAS OPEN RATE" in line:
        #             splitline = line.split()
        #             splitline[4] = str(mu_param[4])
        #             tmp = [i + " " for i in splitline]
        #             new_line = ""
        #             new_line = new_line.join(tmp)
        #             pdb.set_trace()

        file_name = (
            data_folder_root
            + "/fluid/"
            + str(idx_mu)
            + "/case2skew_"
            + str(idx_mu)
            + ".DATA"
        )
        search_pattern = " W1 GAS OPEN RATE 1.3e6 190 /"  # PAY ATTENTION HERE...
        replace_pattern = " W1 GAS OPEN RATE " + str(mu_param[4]) + " 190 /"
        replace_all(file_name, search_pattern, replace_pattern)

        # prepare run working folder
        shutil.copy(
            save_folder_root + "/fluid/" + "FAULTS.INC",
            save_folder_root
            + "/fluid/"
            + str(idx_mu)
            + "/"
            + "FAULTS_"
            + str(idx_mu)
            + ".INC",
        )

        shutil.copy(
            save_folder_root + "/fluid/" + "FIPNUM.INC",
            save_folder_root
            + "/fluid/"
            + str(idx_mu)
            + "/"
            + "FIPNUM_"
            + str(idx_mu)
            + ".INC",
        )
        shutil.copy(
            save_folder_root + "/fluid/" + "FLUXNUM.INC",
            save_folder_root
            + "/fluid/"
            + str(idx_mu)
            + "/"
            + "FLUXNUM_"
            + str(idx_mu)
            + ".INC",
        )
        shutil.copy(
            save_folder_root + "/fluid/" + "PERMX.INC",
            save_folder_root
            + "/fluid/"
            + str(idx_mu)
            + "/"
            + "PERMX_"
            + str(idx_mu)
            + ".INC",
        )
        shutil.copy(
            save_folder_root + "/fluid/" + "PERMY.INC",
            save_folder_root
            + "/fluid/"
            + str(idx_mu)
            + "/"
            + "PERMY_"
            + str(idx_mu)
            + ".INC",
        )
        shutil.copy(
            save_folder_root + "/fluid/" + "PERMZ.INC",
            save_folder_root
            + "/fluid/"
            + str(idx_mu)
            + "/"
            + "PERMZ_"
            + str(idx_mu)
            + ".INC",
        )
        shutil.copy(
            save_folder_root + "/fluid/" + "PORO.INC",
            save_folder_root
            + "/fluid/"
            + str(idx_mu)
            + "/"
            + "PORO_"
            + str(idx_mu)
            + ".INC",
        )

        shutil.copy(
            save_folder_root + "/fluid/" + "PUNQS3_EOS_COMPS.INC",
            save_folder_root
            + "/fluid/"
            + str(idx_mu)
            + "/"
            + "PUNQS3_EOS_COMPS_"
            + str(idx_mu)
            + ".INC",
        )

        shutil.copy(
            save_folder_root + "/fluid/" + "ROCK.INC",
            save_folder_root
            + "/fluid/"
            + str(idx_mu)
            + "/"
            + "ROCK_"
            + str(idx_mu)
            + ".INC",
        )

        shutil.copy(
            save_folder_root + "/fluid/" + "ROCKNUM.INC",
            save_folder_root
            + "/fluid/"
            + str(idx_mu)
            + "/"
            + "ROCKNUM_"
            + str(idx_mu)
            + ".INC",
        )

        shutil.copy(
            save_folder_root + "/fluid/" + "SCAL.INC",
            save_folder_root
            + "/fluid/"
            + str(idx_mu)
            + "/"
            + "SCAL_"
            + str(idx_mu)
            + ".INC",
        )

        shutil.copy(
            save_folder_root + "/fluid/" + "case2skew.EGRID",
            save_folder_root
            + "/fluid/"
            + str(idx_mu)
            + "/"
            + "case2skew_"
            + str(idx_mu)
            + ".INC",
        )

        pdb.set_trace()

        # run echelon here? no clue should I submit a lot of files?

        # fetch results here?

        # convert, see read_unrst
