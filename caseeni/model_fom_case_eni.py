import os
import sys
import pdb
import warnings
import inspect
import copy

# sys.path.append("/home/inspiron/Desktop/PhD/eni_venv/porepy/src")

import numpy as np
import scipy as sp
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
        save_folder: str,
        idx_mu: int,
    ) -> None:
        """ """

        pp_model, pp_params = self.specific_pp_model(save_folder, mu_param, idx_mu)
        pp.run_time_dependent_model(pp_model, pp_params)

        del pp_model, pp_params

    def specific_pp_model(self, save_folder_root, mu_param, idx_mu):
        """ """
        save_folder = save_folder_root + "/" + str(idx_mu)

        pp_model = sub_model_fom_case_eni.ModelCaseEni()

        pp_model.save_folder = save_folder
        pp_model.mu_param = mu_param
        pp_model.exporter_folder = save_folder

        pp_params = {}

        return pp_model, pp_params
