import scipy as sp
import numpy as np
from typing import Callable, Optional, Type, Literal, Sequence, Union
import porepy as pp
import pygem
import copy
import os
import pdb
import warnings
import porepy.models.two_phase_hu as two_phase_hu
from flow_benchmark_3d import _flow_3d

"""


"""


class SolutionStrategyCase4(two_phase_hu.SolutionStrategyPressureMass):
    def prepare_simulation(self) -> None:
        """ """
        self.clean_working_directory()

        self.set_geometry()

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

        self.save_data_time_step()

        self.computations_for_hu()

    def clean_working_directory(self):
        """ """
        os.system("rm " + self.output_file_name)
        os.system("rm " + self.mass_output_file_name)
        os.system("rm " + self.flips_file_name)
        os.system("rm " + self.beta_file_name)

    def initial_condition(self) -> None:
        """ """
        val = np.zeros(self.equation_system.num_dofs())
        for time_step_index in self.time_step_indices:
            self.equation_system.set_variable_values(
                val,
                time_step_index=time_step_index,
            )

        for iterate_index in self.iterate_indices:
            self.equation_system.set_variable_values(val, iterate_index=iterate_index)

        for sd in self.mdg.subdomains():
            saturation_variable = (
                self.mixture.mixture_for_subdomain(sd)
                .get_phase(self.ell)
                .saturation_operator([sd])
            )

            saturation_values = 0.0 * np.ones(sd.num_cells)
            saturation_values[np.where(sd.cell_centers[2] >= 0.75 * self.zmax)] = 1.0

            # if sd.dim == 1:
            #     saturation_values = 0.5 * np.ones(sd.num_cells)

            self.equation_system.set_variable_values(
                saturation_values,
                variables=[saturation_variable],
                time_step_index=0,
                iterate_index=0,
            )

            pressure_variable = self.pressure([sd])

            pressure_values = (2 - 1 * sd.cell_centers[2] / self.zmax) / self.p_0

            self.equation_system.set_variable_values(
                pressure_values,
                variables=[pressure_variable],
                time_step_index=0,
                iterate_index=0,
            )

        for sd, data in self.mdg.subdomains(return_data=True):
            pp.initialize_data(
                sd,
                data,
                self.ppu_keyword,
                {
                    "darcy_flux_phase_0": np.zeros(sd.num_faces),
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

    # def after_nonlinear_iteration(self, solution_vector: np.ndarray) -> None:
    #     """ """
    #     self._nonlinear_iteration += 1
    #     self.equation_system.shift_iterate_values()
    #     self.equation_system.set_variable_values(
    #         values=solution_vector, additive=True, iterate_index=0
    #     )
    #     self.clip_saturation()

    #     # for sd in self.mdg.subdomains():
    #     #     print("sd.dim = ", sd.dim)

    #     np.set_printoptions(precision=16)
    #     a = self.aperture(self.mdg.subdomains())
    #     a = a.evaluate(self.equation_system)
    #     print("a = ", a)
    #     pdb.set_trace()

    def eb_after_timestep(self):
        """used for plots, tests, and more"""

        self.compute_mass()
        self.beta_faces_model(dim_to_check=self.mdg.dim_max())

        # pdb.set_trace()
        # ka = self.intrinsic_permeability([self.mdg.subdomains(dim=3)[0]]).evaluate(
        #     self.equation_system
        # )


class ConstitutiveLawCase4(
    pp.constitutive_laws.DimensionReduction,
):
    def intrinsic_permeability(self, subdomains: list[pp.Grid]) -> pp.ad.Operator:
        """for some reason subdoamins is always a single domain"""

        sd = subdomains[0]

        # if sd.dim == 0:
        #     extra_pts = sd.cell_centers.T
        # else:
        #     extra_pts = sd.nodes.T

        # print("\n\n\n id = ", sd.id)
        # sd_3d = self.mdg.subdomains(dim=3)[0]
        # pp.plot_grid(sd_3d, extra_pts=extra_pts, alpha=0)

        if len(subdomains) > 1:
            print("\n\n\n check intrinsic_permeability\n")
            raise NotImplementedError

        if sd.dim == 3:
            c = sd.cell_centers
            idx_region_1 = [
                np.where(np.logical_and(c[0] > 0.5, c[1] < 0.5))[0],
                np.where(
                    np.logical_and(
                        np.logical_and(
                            c[0] > 0.75, np.logical_and(c[1] > 0.5, c[1] < 0.75)
                        ),
                        c[2] > 0.5,
                    )
                )[0],
                np.where(
                    np.logical_and(
                        np.logical_and(
                            np.logical_and(c[0] > 0.625, c[0] < 0.75),
                            np.logical_and(c[1] > 0.5, c[1] < 0.625),
                        ),
                        np.logical_and(c[2] > 0.5, c[2] < 0.75),
                    )
                )[0],
            ]
            idx_region_1 = np.sort(
                np.concatenate(idx_region_1)
            )  # sort to make it readable

            idx_region_0 = np.setdiff1d(np.arange(sd.num_cells), idx_region_1)

            values = np.zeros(sd.num_cells)
            values[idx_region_0] = 1.0
            values[idx_region_1] = 1e-1
            permeability = pp.ad.DenseArray(values)

            # sd_3d = self.mdg.subdomains(dim=3)[0]
            # # extra_pts = c.T[idx_region_0]
            # # pp.plot_grid(sd_3d, extra_pts=extra_pts, alpha=0)

            # exporter = pp.Exporter(sd_3d, "./case_4/check_K")
            # exporter.write_vtu(("K", values))  # exporter is nosense

        elif sd.dim == 2:
            permeability = pp.ad.DenseArray(1e0 * np.ones(sd.num_cells))
        elif sd.dim == 1:
            permeability = pp.ad.DenseArray(1e-4 * np.ones(sd.num_cells))
        else:  # 0D
            warnings.warn("setting intrinsic permeability to 0D grid")
            permeability = pp.ad.DenseArray(np.ones(sd.num_cells))  # do i need it?

        permeability.set_name("intrinsic_permeability")
        return permeability

    def normal_perm(self, interfaces) -> pp.ad.Operator:
        """ """

        perm = [None] * len(interfaces)

        for id_intf, intf in enumerate(interfaces):
            # print("\n\n\n id = ", intf.id)
            # sd_2d = self.mdg.subdomains(dim=2)[0]
            # pp.plot_grid(sd_2d, extra_pts=intf.cell_centers.T, alpha=0)

            if intf.dim == 2:
                perm[id_intf] = 2e8 * np.ones([intf.num_cells])

            elif intf.dim == 1:
                perm[id_intf] = 2e4 * np.ones([intf.num_cells])
            else:  # 0D
                perm[id_intf] = 2e0 * np.ones([intf.num_cells])

        norm_perm = pp.ad.DenseArray(np.concatenate(perm))

        return norm_perm

    def porosity(self, subdomains: list[pp.Grid]) -> pp.ad.Operator:
        """ """
        phi = [None] * len(subdomains)

        for index, sd in enumerate(subdomains):
            if sd.dim == 3:
                phi[index] = 1e-1 * np.ones([sd.num_cells])
            if sd.dim == 2:
                phi[index] = 9e-1 * np.ones([sd.num_cells])
            if sd.dim == 1:
                phi[index] = 9e-1 * np.ones([sd.num_cells])
            if sd.dim == 0:
                warnings.warn("setting porosity to 0D grid")
                phi[index] = 1.0 * np.ones([sd.num_cells])

        return pp.ad.DenseArray(np.concatenate(phi))

    def grid_aperture(self, sd: pp.Grid) -> np.ndarray:
        """ """
        aperture = np.ones(sd.num_cells)
        residual_aperture_by_dim = [
            1e-12,
            1e-8,
            1e-4,
            1.0,
        ]  # 0D, 1D, 2D, 3D
        aperture = residual_aperture_by_dim[sd.dim] * aperture
        return aperture


class GeometryCase3(pp.ModelGeometry):
    def set_geometry(self) -> None:
        """ """

        self.set_domain()
        self.set_fractures()

        self.fracture_network = pp.create_fracture_network(self.fractures, self.domain)

        self.mdg = pp.create_mdg(
            "cartesian",
            self.meshing_arguments(),
            self.fracture_network,
            **self.meshing_kwargs(),
        )
        self.nd: int = self.mdg.dim_max()

        self.mdg.compute_geometry()

        # exporter = pp.Exporter(self.mdg, "mdg_I_hope", "./case_4/")
        # exporter.write_pvd()
        # exporter.write_vtu()

        print(self.mdg)
        print("\n\n")

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

        # 0,0,0,1,1,1
        # 0.5,0,0,0.5,1,0,0.5,1,1,0.5,0,1
        # 0,0.5,0,1,0.5,0,1,0.5,1,0,0.5,1
        # 0,0,0.5,1,0,0.5,1,1,0.5,0,1,0.5
        # 0.75,0.5,0.5,0.75,1.0,0.5,0.75,1.0,1.0,0.75,0.5,1.0
        # 0.5,0.5,0.75,1.0,0.5,0.75,1.0,1.0,0.75,0.5,1.0,0.75
        # 0.5,0.75,0.5,1.0,0.75,0.5,1.0,0.75,1.0,0.5,0.75,1.0
        # 0.50,0.625,0.50,0.75,0.625,0.50,0.75,0.625,0.75,0.50,0.625,0.75
        # 0.625,0.50,0.50,0.625,0.75,0.50,0.625,0.75,0.75,0.625,0.50,0.75
        # 0.50,0.50,0.625,0.75,0.50,0.625,0.75,0.75,0.625,0.50,0.75,0.625

        pts_0 = np.array([[0.5, 0, 0], [0.5, 1, 0], [0.5, 1, 1], [0.5, 0, 1]])
        frac_0 = pp.PlaneFracture(pts_0.T)

        pts_1 = np.array([[0, 0.5, 0], [1, 0.5, 0], [1, 0.5, 1], [0, 0.5, 1]])
        frac_1 = pp.PlaneFracture(pts_1.T)

        pts_2 = np.array([[0, 0, 0.5], [1, 0, 0.5], [1, 1, 0.5], [0, 1, 0.5]])
        frac_2 = pp.PlaneFracture(pts_2.T)

        pts_3 = np.array(
            [[0.75, 0.5, 0.5], [0.75, 1.0, 0.5], [0.75, 1.0, 1.0], [0.75, 0.5, 1.0]]
        )
        frac_3 = pp.PlaneFracture(pts_3.T)

        pts_4 = np.array(
            [[0.5, 0.5, 0.75], [1.0, 0.5, 0.75], [1.0, 1.0, 0.75], [0.5, 1.0, 0.75]]
        )
        frac_4 = pp.PlaneFracture(pts_4.T)

        pts_5 = np.array(
            [[0.5, 0.75, 0.5], [1.0, 0.75, 0.5], [1.0, 0.75, 1.0], [0.5, 0.75, 1.0]]
        )
        frac_5 = pp.PlaneFracture(pts_5.T)

        pts_6 = np.array(
            [
                [0.50, 0.625, 0.50],
                [0.75, 0.625, 0.50],
                [0.75, 0.625, 0.75],
                [0.50, 0.625, 0.75],
            ]
        )
        frac_6 = pp.PlaneFracture(pts_6.T)

        pts_7 = np.array(
            [
                [0.625, 0.50, 0.50],
                [0.625, 0.75, 0.50],
                [0.625, 0.75, 0.75],
                [0.625, 0.50, 0.75],
            ]
        )
        frac_7 = pp.PlaneFracture(pts_7.T)

        pts_8 = np.array(
            [
                [0.50, 0.50, 0.625],
                [0.75, 0.50, 0.625],
                [0.75, 0.75, 0.625],
                [0.50, 0.75, 0.625],
            ]
        )
        frac_8 = pp.PlaneFracture(pts_8.T)

        self._fractures: list = [
            frac_0,
        ]
        # frac_1,
        # frac_2,
        # frac_3,
        # frac_4,
        #     frac_5,
        #     frac_6,
        #     frac_7,
        #     frac_8,
        # ]

    def meshing_arguments(self) -> dict[str, float]:
        """ """
        default_meshing_args: dict[str, float] = {
            "cell_size": 0.1,
            "cell_size_fracture": 0.1,
        }
        return self.params.get("meshing_arguments", default_meshing_args)


class PartialFinalModel(
    two_phase_hu.PrimaryVariables,
    two_phase_hu.Equations,
    ConstitutiveLawCase4,
    two_phase_hu.BoundaryConditionsPressureMass,
    SolutionStrategyCase4,
    GeometryCase3,
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
            self.ymin = 0.0 / self.L_0
            self.ymax = 1.0 / self.L_0
            self.zmin = 0.0 / self.L_0
            self.zmax = 1.0 / self.L_0

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

            self.root_path = "./case_4/hu/"

            self.output_file_name = self.root_path + "OUTPUT_NEWTON_INFO"
            self.mass_output_file_name = self.root_path + "MASS_OVER_TIME"
            self.flips_file_name = self.root_path + "FLIPS"
            self.beta_file_name = self.root_path + "BETA/BETA"

    os.system("mkdir -p ./case_4/hu/")
    os.system("mkdir -p ./case_4/hu/BETA")
    folder_name = "./case_4/hu/visualization"

    time_manager = two_phase_hu.TimeManagerPP(
        schedule=np.array([0, 1]) / t_0,
        dt_init=1e-3 / t_0,
        dt_min_max=np.array([1e-8, 1e-3]) / t_0,
        constant_dt=False,
        recomp_factor=0.5,
        recomp_max=10,
        iter_max=10,
        print_info=True,
        folder_name=folder_name,
    )

    meshing_kwargs = {"constraints": np.array([20])}
    params = {
        "material_constants": material_constants,
        "max_iterations": 20,
        "nl_convergence_tol": 1e-6,
        "nl_divergence_tol": 1e6,
        "time_manager": time_manager,
        "folder_name": folder_name,
        "meshing_kwargs": meshing_kwargs,
    }

    model = FinalModel(params)

    pp.run_time_dependent_model(model, params)
