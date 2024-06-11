import sys
import scipy as sp
import numpy as np
from typing import Callable, Optional, Type, Literal, Sequence, Union

pp_path = "../../../src"
if pp_path not in sys.path:
    sys.path.append(pp_path)
import porepy as pp

import pygem
import os
import copy
import pdb
import porepy.models.two_phase_hu as two_phase_hu
import porepy.models.two_phase_ppu as two_phase_ppu

import case_1_horizontal_hu
import case_1_horizontal_ppu
import case_1_slanted_hu

"""
- non conforming displacement_max = -0.1, only for permeable fracture
- linear solver is the bottleneck for small mesh size

"""


class InitialConditionCase1SlantedPPU(
    case_1_slanted_hu.InitialConditionCase1Slanted
):  # I suck in python
    def initial_condition(self) -> None:
        """ """
        self.initial_condition_common()

        for sd, data in self.mdg.subdomains(return_data=True):
            pp.initialize_data(
                sd,
                data,
                self.ppu_keyword + "_" + "darcy_flux_phase_0",
                {
                    "darcy_flux_phase_0": np.zeros(sd.num_faces),
                },
            )

            pp.initialize_data(
                sd,
                data,
                self.ppu_keyword + "_" + "darcy_flux_phase_1",
                {
                    "darcy_flux_phase_1": np.zeros(sd.num_faces),
                },
            )

        for intf, data in self.mdg.interfaces(return_data=True):
            pp.initialize_data(
                intf,
                data,
                self.ppu_keyword + "_" + "interface_mortar_flux_phase_0",
                {
                    "interface_mortar_flux_phase_0": np.zeros(intf.num_cells),
                },
            )

            pp.initialize_data(
                intf,
                data,
                self.ppu_keyword + "_" + "interface_mortar_flux_phase_1",
                {
                    "interface_mortar_flux_phase_1": np.zeros(intf.num_cells),
                },
            )


class SolutionStrategyCase1SlantedPPU(two_phase_ppu.SolutionStrategyPressureMassPPU):

    def prepare_simulation(self) -> None:
        """ """
        self.deform_grid = case_1_slanted_hu.SolutionStrategyCase1Slanted.deform_grid
        self.compute_normals_tangents = (
            case_1_slanted_hu.SolutionStrategyCase1Slanted.compute_normals_tangents
        )

        self.clean_working_directory()

        self.set_geometry(mdg_ref=False)

        self.deform_grid(self)

        self.initialize_data_saving()

        self.set_materials()

        self.set_equation_system_manager()

        self.add_equation_system_to_phases()
        self.mixture.apply_constraint(self.ell)

        self.create_variables()

        self.initial_condition()
        self.compute_mass()

        self.reset_state_from_file()
        self.set_equations()

        self.set_discretization_parameters()
        self.discretize()

        self._initialize_linear_solver()
        self.set_nonlinear_discretizations()

        self.save_data_time_step()  # it is in pp.viz.data_saving_model_mixin

        self.computations_for_hu()

        # pp.plot_grid(self.mdg, alpha=0)

    def flip_flop(self, dim_to_check):
        """ """
        for sd in self.mdg.subdomains():
            if sd.dim == 2:  # sorry...
                darcy_phase_0 = (
                    self.darcy_flux_phase_0([sd], self.mixture.get_phase(0))
                    .evaluate(self.equation_system)
                    .val
                )
                darcy_phase_1 = (
                    self.darcy_flux_phase_1([sd], self.mixture.get_phase(1))
                    .evaluate(self.equation_system)
                    .val
                )

                sign_darcy_phase_0_2d = np.sign(darcy_phase_0)
                sign_darcy_phase_1_2d = np.sign(darcy_phase_1)

                if self.sign_darcy_phase_0_2d_prev is None:
                    self.sign_darcy_phase_0_2d_prev = sign_darcy_phase_0_2d

                if self.sign_darcy_phase_1_2d_prev is None:
                    self.sign_darcy_phase_1_2d_prev = sign_darcy_phase_1_2d

                number_flips_darcy_phase_0_2d = np.sum(
                    np.not_equal(self.sign_darcy_phase_0_2d_prev, sign_darcy_phase_0_2d)
                )

                number_flips_darcy_phase_1_2d = np.sum(
                    np.not_equal(self.sign_darcy_phase_1_2d_prev, sign_darcy_phase_1_2d)
                )

            if sd.dim == 1:  # sorry...
                darcy_phase_0 = (
                    self.darcy_flux_phase_0([sd], self.mixture.get_phase(0))
                    .evaluate(self.equation_system)
                    .val
                )
                darcy_phase_1 = (
                    self.darcy_flux_phase_1([sd], self.mixture.get_phase(1))
                    .evaluate(self.equation_system)
                    .val
                )

                sign_darcy_phase_0_1d = np.sign(darcy_phase_0)
                sign_darcy_phase_1_1d = np.sign(darcy_phase_1)

                if self.sign_darcy_phase_0_1d_prev is None:
                    self.sign_darcy_phase_0_1d_prev = sign_darcy_phase_0_1d

                if self.sign_darcy_phase_1_1d_prev is None:
                    self.sign_darcy_phase_1_1d_prev = sign_darcy_phase_1_1d

                number_flips_darcy_phase_0_1d = np.sum(
                    np.not_equal(self.sign_darcy_phase_0_1d_prev, sign_darcy_phase_0_1d)
                )

                number_flips_darcy_phase_1_1d = np.sum(
                    np.not_equal(self.sign_darcy_phase_1_1d_prev, sign_darcy_phase_1_1d)
                )

        for intf, data in self.mdg.interfaces(return_data=True):
            mortar_0 = self.interface_mortar_flux_phase_0([intf]).evaluate(
                self.equation_system
            )
            sign_mortar_0 = np.sign(mortar_0.val)

            if self.sign_mortar_0_prev is None:
                self.sign_mortar_0_prev = sign_mortar_0

            number_flips_mortar_0 = np.sum(
                np.not_equal(self.sign_mortar_0_prev, sign_mortar_0)
            )

            self.sign_mortar_0_prev = sign_mortar_0

            mortar_1 = self.interface_mortar_flux_phase_1([intf]).evaluate(
                self.equation_system
            )
            sign_mortar_1 = np.sign(mortar_1.val)

            if self.sign_mortar_1_prev is None:
                self.sign_mortar_1_prev = sign_mortar_1

            number_flips_mortar_1 = np.sum(
                np.not_equal(self.sign_mortar_1_prev, sign_mortar_1)
            )

            self.sign_mortar_0_prev = sign_mortar_0

        return np.array(
            [
                sign_darcy_phase_0_2d,
                sign_darcy_phase_1_2d,
                sign_darcy_phase_0_1d,
                sign_darcy_phase_1_1d,
                sign_mortar_0,
                sign_mortar_1,
            ]
        ), np.array(
            [
                number_flips_darcy_phase_0_2d,
                number_flips_darcy_phase_1_2d,
                number_flips_darcy_phase_0_1d,
                number_flips_darcy_phase_1_1d,
                number_flips_mortar_0,
                number_flips_mortar_1,
            ]
        )


class PartialFinalModel(
    two_phase_hu.PrimaryVariables,
    two_phase_ppu.EquationsPPU,
    case_1_slanted_hu.ConstitutiveLawCase1Slanted,
    two_phase_hu.BoundaryConditionsPressureMass,
    InitialConditionCase1SlantedPPU,
    SolutionStrategyCase1SlantedPPU,
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

    class FinalModel(PartialFinalModel):
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

            self.number_upwind_dirs = 6
            self.sign_darcy_phase_0_2d_prev = None
            self.sign_darcy_phase_1_2d_prev = None
            self.sign_darcy_phase_0_1d_prev = None
            self.sign_darcy_phase_1_1d_prev = None
            self.sign_mortar_0_prev = None
            self.sign_mortar_1_prev = None

            # self.root_path = "./case_1/slanted_ppu_Kn" + str(Kn) + "/"
            self.root_path = "./case_1/slanted_ppu/non-conforming/"

            self.output_file_name = self.root_path + "OUTPUT_NEWTON_INFO"
            self.mass_output_file_name = self.root_path + "MASS_OVER_TIME"
            self.flips_file_name = self.root_path + "FLIPS"
            self.beta_file_name = self.root_path + "BETA"  # not used

    cell_size = 0.05

    # os.system("mkdir -p ./case_1/slanted_ppu_Kn" + str(Kn))
    os.system("mkdir -p ./case_1/slanted_ppu/non-conforming")

    # folder_name = "./case_1/slanted_ppu_Kn" + str(Kn) + "/visualization"
    folder_name = "./case_1/slanted_ppu/non-conforming/visualization"

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
