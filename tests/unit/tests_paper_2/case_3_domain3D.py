import scipy as sp
import numpy as np
from typing import Callable, Optional, Type, Literal, Sequence, Union
import porepy as pp

import os
import sys
import pdb
import warnings

import porepy.models.two_phase_hu as two_phase_hu
import case_3_hu

"""
- dont look at beta files, beta function inside the model is hardoced for 2D

"""


class SolutionStrategyCase3Domain(two_phase_hu.SolutionStrategyPressureMass):
    def before_nonlinear_iteration(self):
        """ """
        sys.exit()


class GeometryCase3Domain(pp.ModelGeometry):
    def set_geometry(self) -> None:
        """ """
        self.set_domain()

        self.set_fractures()
        self.fracture_network = pp.create_fracture_network(self.fractures, self.domain)

        self.mdg = pp.create_mdg(
            "simplex",
            self.meshing_arguments(),
            self.fracture_network,
            **self.meshing_kwargs(),
        )
        self.nd: int = self.mdg.dim_max()

        exporter = pp.Exporter(self.mdg, "mdg_picture", "./case_3/domain")
        exporter.write_pvd()
        exporter.write_vtu()

        self.nd: int = self.mdg.dim_max()
        # pp.set_local_coordinate_projections(self.mdg) # dont know what is this, if it return an error uncomment it...

    def set_domain(self) -> None:
        """
        TODO: you dont really need this set_domain... you know
        """
        bounding_box = {
            "xmin": self.xmin,
            "xmax": self.xmax,
            "ymin": self.ymin,
            "ymax": self.ymax,
            "zmin": self.zmin,
            "zmax": self.zmax,
        }
        self._domain = pp.Domain(bounding_box=bounding_box)

    def set_fractures(self) -> None:
        """ """

        # R = pp.map_geometry.rotation_matrix(np.pi / 2, np.array([1, 0, 0]))

        # pts_0 = np.array(
        #     [[0.05, 0.25, 0.5], [0.95, 0.25, 0.5], [0.95, 2, 0.5], [0.05, 2, 0.5]]
        # )
        # frac_0 = pp.PlaneFracture(R @ pts_0.T)

        # pts_1 = np.array(
        #     [[0.5, 0.05, 0.95], [0.5, 0.05, 0.05], [0.5, 0.3, 0.05], [0.5, 0.3, 0.95]]
        # )
        # frac_1 = pp.PlaneFracture(R @ pts_1.T)

        # pts_2 = np.array(
        #     [[0.05, 1, 0.5], [0.95, 1, 0.5], [0.95, 2.2, 0.85], [0.05, 2.2, 0.85]]
        # )
        # frac_2 = pp.PlaneFracture(R @ pts_2.T)

        # pts_3 = np.array(
        #     [[0.05, 1, 0.48], [0.95, 1, 0.48], [0.95, 2.2, 0.14], [0.05, 2.2, 0.14]]
        # )
        # frac_3 = pp.PlaneFracture(R @ pts_3.T)

        # pts_4 = np.array(
        #     [[0.23, 1.9, 0.3], [0.23, 1.9, 0.7], [0.17, 2.2, 0.7], [0.17, 2.2, 0.3]]
        # )
        # frac_4 = pp.PlaneFracture(R @ pts_4.T)

        # pts_5 = np.array(
        #     [[0.17, 1.9, 0.3], [0.17, 1.9, 0.7], [0.23, 2.2, 0.7], [0.23, 2.2, 0.3]]
        # )
        # frac_5 = pp.PlaneFracture(R @ pts_5.T)

        # pts_6 = np.array(
        #     [[0.77, 1.9, 0.3], [0.77, 1.9, 0.7], [0.77, 2.2, 0.7], [0.77, 2.2, 0.3]]
        # )
        # frac_6 = pp.PlaneFracture(R @ pts_6.T)

        # pts_7 = np.array(
        #     [[0.83, 1.9, 0.3], [0.83, 1.9, 0.7], [0.83, 2.2, 0.7], [0.83, 2.2, 0.3]]
        # )
        # frac_7 = pp.PlaneFracture(R @ pts_7.T)

        # frac_8_constr = pp.PlaneFracture(
        #     R
        #     @ np.array(
        #         [
        #             [0.0, self.z_cut, 0],  # it will be a z...
        #             [1, self.z_cut, 0],
        #             [1, self.z_cut, 1],
        #             [0, self.z_cut, 1],
        #         ]
        #     ).T
        # )

        # self._fractures: list = [
        #     frac_0,
        #     frac_1,
        #     frac_2,
        #     frac_3,
        #     frac_4,
        #     frac_5,
        #     frac_6,
        #     frac_7,
        #     frac_8_constr,
        # ]

        self._fractures = []

    def meshing_arguments(self) -> dict[str, float]:
        """ """
        default_meshing_args: dict[str, float] = {
            "cell_size": 0.3 / self.L_0,
            "cell_size_fracture": 0.2 / self.L_0,
        }
        return self.params.get("meshing_arguments", default_meshing_args)


class PartialFinalModel(
    two_phase_hu.PrimaryVariables,
    two_phase_hu.Equations,
    case_3_hu.ConstitutiveLawCase3,
    two_phase_hu.BoundaryConditionsPressureMass,
    case_3_hu.InitialConditionCase3,
    SolutionStrategyCase3Domain,
    GeometryCase3Domain,
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

            self.mdg = None
            self.cell_size = None

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
            self.ymin = -1.0 / self.L_0
            self.ymax = 0.0 / self.L_0
            self.zmin = 0.0 / self.L_0
            self.zmax = 2.25 / self.L_0
            self.z_cut = 0.8

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

            self.root_path = "./case_3/hu/"

            self.output_file_name = self.root_path + "OUTPUT_NEWTON_INFO"
            self.mass_output_file_name = self.root_path + "MASS_OVER_TIME"
            self.flips_file_name = self.root_path + "FLIPS"
            self.beta_file_name = self.root_path + "BETA"

    # os.system("mkdir -p ./case_3/hu/")
    # os.system("mkdir -p ./case_3/hu/BETA")
    os.system("mkdir ./case_3/domain")
    folder_name = "./case_3/hu/visualization"

    time_manager = two_phase_hu.TimeManagerPP(
        schedule=np.array([0, 100]) / t_0,
        dt_init=1e-1 / t_0,
        dt_min_max=np.array([1e-9, 1e-1]) / t_0,
        constant_dt=False,
        recomp_factor=0.5,
        recomp_max=10,
        iter_max=10,
        print_info=True,
        folder_name=folder_name,
    )

    meshing_kwargs = {"constraints": np.array([8])}
    params = {
        "material_constants": material_constants,
        "max_iterations": 15,
        "nl_convergence_tol": 2e-5,
        "nl_divergence_tol": 1e5,
        "time_manager": time_manager,
        "folder_name": folder_name,
        "meshing_kwargs": meshing_kwargs,
    }

    model = FinalModel(params)

    pp.run_time_dependent_model(model, params)
