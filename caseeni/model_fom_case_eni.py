import os
import sys
import pdb
import warnings
import inspect
import copy
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


class ModelCaseEni:
    def __init__(self, data_folder):
        """ """
        self.data_folder = data_folder

    def solve_one_instance_ti_tf(
        self,
        mu_param: np.array,
        save_folder_root: str,
        idx_mu: int,
    ) -> None:
        """ """
        times = np.loadtxt(save_folder_root + "/TIMES")
        echelon_pressure = np.load(
            save_folder_root + "/fluid/pressure_" + str(idx_mu) + ".npy"
        )
        save_folder = save_folder_root + "/mech/" + str(idx_mu)

        pp_model = sub_model_fom_case_eni.SubModelCaseEni()
        pp_model.save_folder = save_folder
        pp_model.exporter_folder = save_folder

        for time in times:
            pp_model.subscript = "_" + str(time)
            pp_model.mu_param = mu_param
            pp_model.echelon_pressure = None  # echelon_pressure
            pp_params = {}
            pp.run_stationary_model(pp_model, pp_params)

        del pp_model, pp_params
