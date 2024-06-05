import scipy as sp
import numpy as np
from typing import Callable, Optional, Type, Literal, Sequence, Union

import sys

pp_path = "../../../src"
if pp_path not in sys.path:
    sys.path.append(pp_path)
import porepy as pp


import pygem
import copy
import os
import pdb
import warnings

import porepy.models.two_phase_hu as two_phase_hu
from flow_benchmark_3d import _flow_3d

"""

- THIS is case 3 with smaller timestep. case 3 had dt = 3e-3. now (19/03/2024) case 3 has dt = 1e-3 so case 3 = case 8

"""


class InitialConditionCase3:
    def initial_condition_common(self) -> None:
        """
        common to hu and ppu
        """
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
            saturation_values[np.where(sd.cell_centers[2] >= self.z_cut)] = 1.0

            # if (sd.dim == 2) or (sd.dim == 1):
            #     saturation_values = 1e-6 * np.ones(sd.num_cells)

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

    def initial_condition(self) -> None:
        """ """
        self.initial_condition_common()

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


class ConstitutiveLawCase3(
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
            print("\n\n\n check intrinsic_permeability")
            raise NotImplementedError

        if sd.dim == 3:
            permeability = pp.ad.DenseArray(1e2 / 1e4 * np.ones(sd.num_cells))
        elif sd.dim == 2:
            permeability = pp.ad.DenseArray(1e4 / 1e4 * np.ones(sd.num_cells))
        elif sd.dim == 1:
            permeability = pp.ad.DenseArray(1e4 / 1e4 * np.ones(sd.num_cells))
        else:  # 0D
            warnings.warn("setting intrinsic permeability to 0D grid")
            permeability = pp.ad.DenseArray(np.ones(sd.num_cells))

        permeability.set_name("intrinsic_permeability")
        return permeability

    def intrinsic_permeability_tensor(self, sd: pp.Grid) -> pp.SecondOrderTensor:
        """ """
        permeability_ad = self.specific_volume([sd]) * self.intrinsic_permeability([sd])
        try:
            permeability = permeability_ad.evaluate(self.equation_system)
        except KeyError:
            volume = self.specific_volume([sd]).evaluate(self.equation_system)
            permeability = self.solid.permeability() * np.ones(sd.num_cells) * volume
        if isinstance(permeability, pp.ad.AdArray):
            permeability = permeability.val
        return pp.SecondOrderTensor(permeability)

    def normal_perm(self, interfaces) -> pp.ad.Operator:
        """ """

        perm = [None] * len(interfaces)

        for id_intf, intf in enumerate(interfaces):
            # print("\n\n\n id = ", intf.id)
            # sd_2d = self.mdg.subdomains(dim=2)[0]
            # pp.plot_grid(sd_2d, extra_pts=intf.cell_centers.T, alpha=0)

            if intf.dim == 2:
                perm[id_intf] = 1e4 / 1e4 * np.ones([intf.num_cells])

            elif intf.dim == 1:
                perm[id_intf] = 1e4 / 1e4 * np.ones([intf.num_cells])
            else:  # 0D
                warnings.warn("setting normal permeability to 0D interface")
                perm[id_intf] = np.ones([intf.num_cells])

        norm_perm = pp.ad.DenseArray(np.concatenate(perm))

        return norm_perm

    def porosity(self, subdomains: list[pp.Grid]) -> pp.ad.Operator:
        """ """
        phi = [None] * len(subdomains)

        for index, sd in enumerate(subdomains):
            if sd.dim == 3:
                phi[index] = 2e-1 * np.ones([sd.num_cells])
            if sd.dim == 2:
                phi[index] = 2e-1 * np.ones([sd.num_cells])
            if sd.dim == 1:
                phi[index] = 2e-1 * np.ones([sd.num_cells])
            if sd.dim == 0:
                warnings.warn("setting porosity to 0D grid")
                phi[index] = 2e-1 * np.ones([sd.num_cells])

        return pp.ad.DenseArray(np.concatenate(phi))

    def grid_aperture(self, sd: pp.Grid) -> np.ndarray:
        """ """
        aperture = np.ones(sd.num_cells)
        residual_aperture_by_dim = [
            1.0,
            0.01,
            0.01,
            1.0,
        ]  # 0D, 1D, 2D, 3D
        if sd.dim == 0:
            warnings.warn("setting residual aperture to 0D grid")
        aperture = residual_aperture_by_dim[sd.dim] * aperture
        return aperture


class GeometryCase3(pp.ModelGeometry):
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

        # exporter = pp.Exporter(self.mdg, "mdg_I_hope", "./case_8/")
        # exporter.write_pvd()
        # exporter.write_vtu()

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
        # 0, 0, 0, 1, 2.25, 1
        # 0.05, 0.25, 0.5, 0.95, 0.25, 0.5, 0.95, 2, 0.5, 0.05, 2, 0.5
        # 0.5, 0.05, 0.95, 0.5, 0.05, 0.05, 0.5, 0.3, 0.05, 0.5, 0.3, 0.95
        # 0.05, 1, 0.5, 0.95, 1, 0.5, 0.95, 2.2, 0.85, 0.05, 2.2, 0.85
        # 0.05, 1, 0.48, 0.95, 1, 0.48, 0.95, 2.2, 0.14, 0.05, 2.2, 0.14
        # 0.23, 1.9, 0.3, 0.23, 1.9, 0.7, 0.17, 2.2, 0.7, 0.17, 2.2, 0.3
        # 0.17, 1.9, 0.3, 0.17, 1.9, 0.7, 0.23, 2.2, 0.7, 0.23, 2.2, 0.3
        # 0.77, 1.9, 0.3, 0.77, 1.9, 0.7, 0.77, 2.2, 0.7, 0.77, 2.2, 0.3
        # 0.83, 1.9, 0.3, 0.83, 1.9, 0.7, 0.83, 2.2, 0.7, 0.83, 2.2, 0.3

        R = pp.map_geometry.rotation_matrix(np.pi / 2, np.array([1, 0, 0]))

        pts_0 = np.array(
            [[0.05, 0.25, 0.5], [0.95, 0.25, 0.5], [0.95, 2, 0.5], [0.05, 2, 0.5]]
        )
        frac_0 = pp.PlaneFracture(R @ pts_0.T)

        pts_1 = np.array(
            [[0.5, 0.05, 0.95], [0.5, 0.05, 0.05], [0.5, 0.3, 0.05], [0.5, 0.3, 0.95]]
        )
        frac_1 = pp.PlaneFracture(R @ pts_1.T)

        pts_2 = np.array(
            [[0.05, 1, 0.5], [0.95, 1, 0.5], [0.95, 2.2, 0.85], [0.05, 2.2, 0.85]]
        )
        frac_2 = pp.PlaneFracture(R @ pts_2.T)

        pts_3 = np.array(
            [[0.05, 1, 0.48], [0.95, 1, 0.48], [0.95, 2.2, 0.14], [0.05, 2.2, 0.14]]
        )
        frac_3 = pp.PlaneFracture(R @ pts_3.T)

        pts_4 = np.array(
            [[0.23, 1.9, 0.3], [0.23, 1.9, 0.7], [0.17, 2.2, 0.7], [0.17, 2.2, 0.3]]
        )
        frac_4 = pp.PlaneFracture(R @ pts_4.T)

        pts_5 = np.array(
            [[0.17, 1.9, 0.3], [0.17, 1.9, 0.7], [0.23, 2.2, 0.7], [0.23, 2.2, 0.3]]
        )
        frac_5 = pp.PlaneFracture(R @ pts_5.T)

        pts_6 = np.array(
            [[0.77, 1.9, 0.3], [0.77, 1.9, 0.7], [0.77, 2.2, 0.7], [0.77, 2.2, 0.3]]
        )
        frac_6 = pp.PlaneFracture(R @ pts_6.T)

        pts_7 = np.array(
            [[0.83, 1.9, 0.3], [0.83, 1.9, 0.7], [0.83, 2.2, 0.7], [0.83, 2.2, 0.3]]
        )
        frac_7 = pp.PlaneFracture(R @ pts_7.T)

        frac_8_constr = pp.PlaneFracture(
            R
            @ np.array(
                [
                    [0.0, self.z_cut, 0],  # it will be a z...
                    [1, self.z_cut, 0],
                    [1, self.z_cut, 1],
                    [0, self.z_cut, 1],
                ]
            ).T
        )

        self._fractures: list = [
            frac_0,
            frac_1,
            frac_2,
            frac_3,
            frac_4,
            frac_5,
            frac_6,
            frac_7,
            frac_8_constr,
        ]

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
    ConstitutiveLawCase3,
    two_phase_hu.BoundaryConditionsPressureMass,
    InitialConditionCase3,
    two_phase_hu.SolutionStrategyPressureMass,
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

            self.root_path = "./case_8/hu/"

            self.output_file_name = self.root_path + "OUTPUT_NEWTON_INFO"
            self.mass_output_file_name = self.root_path + "MASS_OVER_TIME"
            self.flips_file_name = self.root_path + "FLIPS"
            self.beta_file_name = self.root_path + "BETA"

    os.system("mkdir -p ./case_8/hu/")
    os.system("mkdir -p ./case_8/hu/BETA")
    folder_name = "./case_8/hu/visualization"

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
