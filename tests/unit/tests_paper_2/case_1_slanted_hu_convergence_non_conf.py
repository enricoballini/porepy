import scipy as sp
import numpy as np
from typing import Callable, Optional, Type, Literal, Sequence, Union

import sys

pp_path = "../../../src"
if pp_path not in sys.path:
    sys.path.append(pp_path)
import porepy as pp


import os
import copy

import pdb
import porepy.models.two_phase_hu as two_phase_hu

import case_1_horizontal_hu
import case_1_slanted_hu
import case_1_slanted_hu_convergence

"""

"""


class SolutionStrategyCase1SlantedNonConformingConvergence(
    case_1_slanted_hu_convergence.SolutionStrategyCase1SlantedConvergence,
    case_1_slanted_hu.SolutionStrategyCase1Slanted,  # remember that ... is from left to right
):
    def prepare_simulation(self) -> None:
        """ """
        self.clean_working_directory()

        self.set_geometry(mdg_ref=True)
        self.set_geometry(mdg_ref=False)

        self.deform_grid(mdg_ref=True)
        self.deform_grid(mdg_ref=False)

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

    def flip_flop(
        self, dim_to_check
    ):  # eh eh here comes the rub with the mixing... you are not taking flip_flop of SolutionStrategyCase1SlantedConvergence
        """very inefficient, i want to make this function as independent as possible"""

        for sd, data in self.mdg.subdomains(return_data=True):
            if sd.dim == dim_to_check:  # for simplicity...
                # total flux flips:
                pressure_adarray = self.pressure([sd]).evaluate(self.equation_system)
                left_restriction = data["for_hu"]["left_restriction"]
                right_restriction = data["for_hu"]["right_restriction"]
                transmissibility_internal_tpfa = data["for_hu"][
                    "transmissibility_internal_tpfa"
                ]
                ad = True
                dynamic_viscosity = data["for_hu"]["dynamic_viscosity"]
                dim_max = data["for_hu"]["dim_max"]
                total_flux_internal = (
                    pp.numerics.fv.hybrid_weighted_average.total_flux_internal(
                        sd,
                        self.mixture.mixture_for_subdomain(sd),
                        pressure_adarray,
                        self.gravity_value,
                        left_restriction,
                        right_restriction,
                        transmissibility_internal_tpfa,
                        ad,
                        dynamic_viscosity,
                        dim_max,
                        self.mobility,
                        self.relative_permeability,
                    )
                )
                sign_total_flux_internal = np.sign(
                    total_flux_internal[0].val + total_flux_internal[1].val
                )
                if (
                    self.sign_total_flux_internal_prev is None
                ):  # first iteration after initial condition
                    self.sign_total_flux_internal_prev = sign_total_flux_internal

                number_flips_qt = np.sum(
                    np.not_equal(
                        self.sign_total_flux_internal_prev, sign_total_flux_internal
                    )
                )

                self.sign_total_flux_internal_prev = sign_total_flux_internal

                # omega_0 flips:
                z = -sd.cell_centers[dim_max - 1]

                saturation_list = [None] * self.mixture.num_phases
                g_list = [None] * self.mixture.num_phases
                mobility_list = [None] * self.mixture.num_phases

                for phase_id in np.arange(self.mixture.num_phases):
                    saturation = self.mixture.get_phase(phase_id).saturation
                    saturation_list[phase_id] = saturation
                    rho = self.mixture.get_phase(phase_id).mass_density(
                        self.pressure([sd]).evaluate(self.equation_system).val
                    )
                    rho = pp.hu_utils.density_internal_faces(
                        saturation, rho, left_restriction, right_restriction
                    )
                    g_list[phase_id] = pp.hu_utils.g_internal_faces(
                        z, rho, self.gravity_value, left_restriction, right_restriction
                    )
                    mobility_list[phase_id] = self.mobility(
                        saturation, dynamic_viscosity
                    )

                omega_0 = pp.omega(
                    self.mixture.num_phases,
                    0,
                    mobility_list,
                    g_list,
                    left_restriction,
                    right_restriction,
                    ad,
                )

                sign_omega_0 = np.sign(omega_0.val)

                if self.sign_omega_0_prev is None:
                    self.sign_omega_0_prev = sign_omega_0

                number_flips_omega_0 = np.sum(
                    np.not_equal(self.sign_omega_0_prev, sign_omega_0)
                )

                self.sign_omega_0_prev = sign_omega_0

                # omega_1 flips
                omega_1 = pp.omega(
                    self.mixture.num_phases,
                    1,
                    mobility_list,
                    g_list,
                    left_restriction,
                    right_restriction,
                    ad,
                )

                sign_omega_1 = np.sign(omega_1.val)

                if self.sign_omega_1_prev is None:
                    self.sign_omega_1_prev = sign_omega_1

                number_flips_omega_1 = np.sum(
                    np.not_equal(self.sign_omega_1_prev, sign_omega_1)
                )

                self.sign_omega_1_prev = sign_omega_1

        return np.array(
            [sign_total_flux_internal, sign_omega_0, sign_omega_1]
        ), np.array([number_flips_qt, number_flips_omega_0, number_flips_omega_1])


class PartialFinalModel(
    two_phase_hu.PrimaryVariables,
    two_phase_hu.Equations,
    case_1_slanted_hu.ConstitutiveLawCase1Slanted,
    two_phase_hu.BoundaryConditionsPressureMass,
    case_1_slanted_hu_convergence.InitialConditionCase1SlantedConvergence,
    SolutionStrategyCase1SlantedNonConformingConvergence,
    case_1_slanted_hu_convergence.GeometryCase1SlantedConvergence,
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

    Kn = 0.01
    solid_constants = pp.SolidConstants(
        {
            "porosity": 0.25,
            "intrinsic_permeability": 1.0 / Ka_0,
            "normal_permeability": Kn / Ka_0,
            "residual_aperture": 0.01 / L_0,
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

            self.number_upwind_dirs = 3
            self.sign_total_flux_internal_prev = None
            self.sign_omega_0_prev = None
            self.sign_omega_1_prev = None

            self.root_path = "./case_1/slanted_hu/non-conforming/convergence_results/"
            # self.root_path = (
            #     "./case_1/non-conforming/slanted_hu_Kn"
            #     + str(Kn)
            #     + "/convergence_results/"
            # )

            self.output_file_name = self.root_path + "OUTPUT_NEWTON_INFO"
            self.mass_output_file_name = self.root_path + "MASS_OVER_TIME"
            self.flips_file_name = self.root_path + "FLIPS"
            self.beta_file_name = self.root_path + "BETA"

    # in the first submission it was: cell_sizes = np.array([0.2, 0.1, 0.05, 0.025, 0.005])  # last one is the ref value
    cell_sizes = np.array([0.2, 0.1, 0.05, 0.025, 0.0025])

    os.system("mkdir -p ./case_1/slanted_hu/non-conforming/convergence_results")
    # os.system(
    #     "mkdir -p ./case_1/non-conforming/slanted_hu_Kn"
    #     + str(Kn)
    #     + "/convergence_results"
    # )

    os.system("mkdir -p ./case_1/slanted_hu/non-conforming/convergence_results/BETA")

    np.savetxt(
        "./case_1/slanted_hu/non-conforming/convergence_results/cell_sizes",
        cell_sizes,
    )
    # np.savetxt(
    #     "./case_1/non-conforming/slanted_hu_Kn"
    #     + str(Kn)
    #     + "/convergence_results/cell_sizes",
    #     cell_sizes,
    # )

    for cell_size in cell_sizes:
        print(
            "\n\n\ncell_size = ",
            cell_size,
            "==========================================",
        )

        folder_name = (
            "./case_1/slanted_hu/non-conforming/convergence_results/visualization_"
            + str(cell_size)
        )
        # folder_name = (
        #     "./case_1/non-conforming/slanted_hu_Kn"
        #     + str(Kn)
        #     + "/convergence_results/visualization_"
        #     + str(cell_size)
        # )

        time_manager = two_phase_hu.TimeManagerPP(
            schedule=np.array([0, 1e-4]) / t_0,
            dt_init=1e-4 / t_0,
            dt_min_max=np.array([1e-4, 1e-3]) / t_0,
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
        model.cell_size_ref = cell_sizes[-1]

        pp.run_time_dependent_model(model, params)
