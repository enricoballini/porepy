import os
import sys
import scipy as sp
import numpy as np
from typing import Callable, Optional, Type, Literal, Sequence, Union

pp_path = "../../../src"
if pp_path not in sys.path:
    sys.path.append(pp_path)
import porepy as pp

import copy
import pdb

import porepy.models.two_phase_hu as two_phase_hu
import case_1_slanted_hu

os.system("clear")


class ConstitutiveLawCase1SlantedSmallK(case_1_slanted_hu.ConstitutiveLawCase1Slanted):
    """ """

    def intrinsic_permeability(self, subdomains: list[pp.Grid]) -> pp.ad.Operator:
        """for some reason subdoamins is always a single domain"""

        sd = subdomains[0]

        if len(subdomains) > 1:
            print("\n\n\n check intrinsic_permeability\n")
            raise NotImplementedError

        if sd.dim == 3:
            permeability = pp.ad.DenseArray(1 * np.ones(sd.num_cells))
        elif sd.dim == 2:
            permeability = pp.ad.DenseArray(1 * np.ones(sd.num_cells))
        elif sd.dim == 1:
            permeability = pp.ad.DenseArray(1e-4 * np.ones(sd.num_cells))
        else:  # 0D
            permeability = pp.ad.DenseArray(1 * np.ones(sd.num_cells))

        permeability.set_name("intrinsic_permeability")
        return permeability

    def normal_perm(self, interfaces) -> pp.ad.Operator:
        """ """
        perm = [None] * len(interfaces)

        for id_intf, intf in enumerate(interfaces):
            if intf.dim == 2:
                perm[id_intf] = 0.1 * np.ones([intf.num_cells])
            elif intf.dim == 1:
                perm[id_intf] = 1e-6 * np.ones([intf.num_cells])
            else:  # 0D
                perm[id_intf] = 0.1 * np.ones([intf.num_cells])

        norm_perm = pp.ad.DenseArray(np.concatenate(perm))
        return norm_perm


class PartialFinalModel(
    two_phase_hu.PrimaryVariables,
    two_phase_hu.Equations,
    ConstitutiveLawCase1SlantedSmallK,
    two_phase_hu.BoundaryConditionsPressureMass,
    case_1_slanted_hu.InitialConditionCase1Slanted,
    case_1_slanted_hu.SolutionStrategyCase1Slanted,
    case_1_slanted_hu.GeometryCase1Slanted,
    pp.DataSavingMixin,
):
    """ """


if __name__ == "__main__":
    # scaling:
    # very bad logic, improve it...
    L_0 = 1
    gravity_0 = 1
    dynamic_viscosity_0 = 1
    rho_0 = 1  # |rho_phase_0-rho_phase_1|
    p_0 = 1
    Ka_0 = 1
    u_0 = Ka_0 * p_0 / (dynamic_viscosity_0 * L_0)
    t_0 = L_0 / u_0

    gravity_number = Ka_0 * rho_0 * gravity_0 / (dynamic_viscosity_0 * u_0)

    print("\nSCALING: ======================================")
    print("u_0 = ", u_0)
    print("t_0 = ", t_0)
    print("gravity_number = ", gravity_number)
    print(
        "pay attention: gravity number is not influenced by Ka_0 and dynamic_viscosity_0"
    )
    print("=========================================\n")

    fluid_constants = pp.FluidConstants({})

    Kn = None
    solid_constants = pp.SolidConstants(
        {
            "porosity": None,
            "intrinsic_permeability": None,
            "normal_permeability": None,
            "residual_aperture": None,
        }
    )

    material_constants = {"fluid": fluid_constants, "solid": solid_constants}

    wetting_phase = pp.composite.phase.Phase(rho0=1 / rho_0, p0=p_0, beta=1e-10)
    non_wetting_phase = pp.composite.phase.Phase(rho0=0.5 / rho_0, p0=p_0, beta=1e-10)

    mixture = pp.Mixture()
    mixture.add([wetting_phase, non_wetting_phase])

    class FinalModel(case_1_slanted_hu.PartialFinalModel):
        def __init__(self, params: Optional[dict] = None):
            super().__init__(params)

            self.mdg_ref = None  # fine mesh
            self.mdg = None  # coarse mesh
            self.cell_size = None
            self.cell_size_ref = None

            # scaling values: (not all of them are actually used inside model)
            self.L_0 = L_0
            self.gravity_0 = gravity_0
            self.dynamic_viscosity_0 = dynamic_viscosity_0
            self.rho_0 = rho_0
            self.p_0 = p_0
            self.Ka_0 = Ka_0
            self.t_0 = t_0

            self.mixture = mixture
            self.ell = 0
            self.gravity_value = 1.0 / self.gravity_0
            self.dynamic_viscosity = 1.0 / self.dynamic_viscosity_0

            self.xmin = 0.0 / self.L_0
            self.xmax = 1.0 / self.L_0
            self.ymin = 0.0 / self.L_0
            self.ymax = 1.0 / self.L_0

            self.xmean = (self.xmax - self.xmin) / 2
            self.ymean = (self.ymax - self.ymin) / 2

            self.tilt_angle = 30 * np.pi / 180
            self.x_bottom = (
                self.xmean
                - (self.ymax - self.ymin)
                / 2
                * np.sin(self.tilt_angle)
                / np.cos(self.tilt_angle)
                / self.L_0
            )
            self.x_top = (
                self.xmean
                + (self.ymax - self.ymin)
                / 2
                * np.sin(self.tilt_angle)
                / np.cos(self.tilt_angle)
                / self.L_0
            )
            self.displacement_max = -0.1

            self.relative_permeability = (
                pp.tobedefined.relative_permeability.rel_perm_quadratic
            )
            self.mobility = pp.tobedefined.mobility.mobility(self.relative_permeability)
            self.mobility_operator = pp.tobedefined.mobility.mobility_operator(
                self.mobility
            )

            self.number_upwind_dirs = 3
            self.sign_total_flux_internal_prev = None
            self.sign_omega_0_prev = None
            self.sign_omega_1_prev = None

            # self.root_path = "./case_1/slanted_hu_Kn" + str(Kn) + "/"
            self.root_path = "./case_1/slanted_hu_small_k/non-conforming/"

            self.output_file_name = self.root_path + "OUTPUT_NEWTON_INFO"
            self.mass_output_file_name = self.root_path + "MASS_OVER_TIME"
            self.flips_file_name = self.root_path + "FLIPS"
            self.beta_file_name = self.root_path + "BETA"

    cell_size = 0.05

    # os.system("mkdir -p ./case_1/slanted_hu_Kn" + str(Kn))
    os.system("mkdir -p ./case_1/slanted_hu_small_k/non-conforming")

    # os.system("mkdir -p ./case_1/slanted_hu_Kn" + str(Kn) + "/BETA")
    os.system("mkdir -p ./case_1/slanted_hu_small_k/non-conforming/BETA")

    # folder_name = "./case_1/slanted_hu_Kn" + str(Kn) + "/visualization"
    folder_name = "./case_1/slanted_hu_small_k/non-conforming/visualization"

    time_manager = two_phase_hu.TimeManagerPP(
        schedule=np.array([0, 10]) / t_0,
        dt_init=1e-1 / t_0,
        dt_min_max=np.array([1e-5, 1e-1]) / t_0,
        constant_dt=False,
        recomp_factor=0.5,
        recomp_max=10,
        iter_max=10,
        print_info=True,
        folder_name=folder_name,
    )

    meshing_kwargs = {"constraints": np.array([1, 2])}
    params = {
        "material_constants": material_constants,
        "max_iterations": 15,
        "nl_convergence_tol": 1e-6,
        "nl_divergence_tol": 1e0,
        "time_manager": time_manager,
        "folder_name": folder_name,
        "meshing_kwargs": meshing_kwargs,
    }

    model = FinalModel(params)
    model.cell_size = cell_size

    pp.run_time_dependent_model(model, params)
