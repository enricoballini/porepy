import scipy as sp
import numpy as np
from typing import Callable, Optional, Type, Literal, Sequence, Union
import porepy as pp
import pygem
import copy
import os
import pdb
import porepy.models.two_phase_hu as two_phase_hu

import case_1_horizontal_hu

"""

"""


class InitialConditionCase2(case_1_horizontal_hu.InitialConditionCase1Horizontal):
    def initial_condition_common(self) -> None:
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
            saturation_values[np.where(sd.cell_centers[1] >= self.ymax / 2)] = 1.0
            # saturation_values = 1.0 - 1 * sd.cell_centers[1] / self.ymax

            # if sd.dim == 1:
            #     saturation_values = 0.5 * np.ones(sd.num_cells)

            self.equation_system.set_variable_values(
                saturation_values,
                variables=[saturation_variable],
                time_step_index=0,
                iterate_index=0,
            )

            pressure_variable = self.pressure([sd])

            pressure_values = (2 - 1 * sd.cell_centers[1] / self.ymax) / self.p_0

            self.equation_system.set_variable_values(
                pressure_values,
                variables=[pressure_variable],
                time_step_index=0,
                iterate_index=0,
            )


class ConstitutiveLawCase2(
    pp.constitutive_laws.DimensionReduction,
):
    def intrinsic_permeability(self, subdomains: list[pp.Grid]) -> pp.ad.Operator:
        """for some reason subdoamins is always a single domain"""

        # you wish the following, but perm is inside data and read by discretizations...
        # no, it's correct because set_discretization_parameters calls intrinsic_permeability_tensor

        sd = subdomains[0]

        # if sd.dim == 0:
        #     extra_pts = sd.cell_centers.T
        # else:
        #     extra_pts = sd.nodes.T

        # print("\n\n\n id = ", sd.id)
        # sd_2d = self.mdg.subdomains(dim=2)[0]
        # pp.plot_grid(sd_2d, extra_pts=extra_pts, alpha=0)

        if len(subdomains) > 1:
            print("\n\n\n check intrinsic_permeability")
            raise NotImplementedError

        if sd.id in [
            7,
            6,
            10,
            19,
            8,
        ]:  # 1D
            permeability = pp.ad.DenseArray(1e-2 * np.ones(sd.num_cells))
        elif sd.id in [22, 24, 25, 23]:  # 0D
            permeability = pp.ad.DenseArray(
                1e-2 * np.ones(sd.num_cells)
            )  # do i need it?
        else:
            permeability = pp.ad.DenseArray(1e2 * np.ones(sd.num_cells))

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

            if intf.id in [
                5,
                6,
                7,
                18,
                9,
            ]:  # 1D interfaces
                perm[id_intf] = 1e-2 * np.ones([intf.num_cells])

            elif intf.id in [
                24,
                25,
                26,
                27,
                28,
                29,
                30,
                31,
                33,
                34,
                36,
                38,
                39,
                40,
            ]:  # 0D interfaces
                perm[id_intf] = 2 / (1 / 1e-2 + 1 / 1e2) * np.ones([intf.num_cells])

            else:
                perm[id_intf] = 1e2 * np.ones([intf.num_cells])

        norm_perm = pp.ad.DenseArray(np.concatenate(perm))
        return norm_perm

    def porosity(self, subdomains: list[pp.Grid]) -> pp.ad.Operator:
        """ """
        phi = [None] * len(subdomains)

        for index, sd in enumerate(subdomains):
            if sd.dim == 3:
                phi[index] = 0.25 * np.ones([sd.num_cells])
            if sd.dim == 2:
                phi[index] = 0.25 * np.ones([sd.num_cells])
            if sd.dim == 1:
                phi[index] = 0.25 * np.ones([sd.num_cells])
            if sd.dim == 0:
                phi[index] = 0.25 * np.ones([sd.num_cells])

        return pp.ad.DenseArray(np.concatenate(phi))

    def grid_aperture(self, sd: pp.Grid) -> np.ndarray:
        """pay attention this is the grid aperture, not the aperture."""
        aperture = np.ones(sd.num_cells)
        residual_aperture_by_dim = [
            1e-2,
            1e-2,
            1.0,
            1.0,
        ]  # 0D, 1D, 2D, 3D
        aperture = residual_aperture_by_dim[sd.dim] * aperture
        return aperture

    def aperture(self, subdomains: list[pp.Grid]) -> pp.ad.Operator:
        """ """
        if len(subdomains) == 0:
            return pp.wrap_as_ad_array(0, size=0)
        projection = pp.ad.SubdomainProjections(subdomains, dim=1)

        for i, sd in enumerate(subdomains):
            a_loc = pp.wrap_as_ad_array(self.grid_aperture(sd))
            a_glob = projection.cell_prolongation([sd]) @ a_loc
            if i == 0:
                apertures = a_glob
            else:
                apertures += a_glob
        apertures.set_name("aperture")
        return apertures


class GeometryCase2(pp.ModelGeometry):
    def set_geometry(self, mdg_ref=False) -> None:
        """ """

        self.set_domain()

        file_name = "network_anna_split.csv"

        self.fracture_network = pp.fracture_importer.network_2d_from_csv(
            file_name, domain=self._domain
        )

        self.mdg = pp.create_mdg(
            "simplex",
            self.meshing_arguments(),
            self.fracture_network,
            **self.meshing_kwargs(),
        )

        self.nd: int = self.mdg.dim_max()

        pp.set_local_coordinate_projections(self.mdg)

        self.set_well_network()
        if len(self.well_network.wells) > 0:
            assert isinstance(self.fracture_network, pp.FractureNetwork3d)
            pp.compute_well_fracture_intersections(
                self.well_network, self.fracture_network
            )
            self.well_network.mesh(self.mdg)

    def set_domain(self) -> None:
        """ """
        self.size = 1
        bounding_box = {
            "xmin": self.xmin,
            "xmax": self.xmax,
            "ymin": self.ymin,
            "ymax": self.ymax,
        }
        self._domain = pp.Domain(bounding_box=bounding_box)

    def meshing_arguments(self) -> dict[str, float]:
        """ """
        default_meshing_args: dict[str, float] = {
            "cell_size": 0.1 / self.L_0,
            "cell_size_fracture": 0.1 / self.L_0,
        }
        return self.params.get("meshing_arguments", default_meshing_args)


class PartialFinalModel(
    two_phase_hu.PrimaryVariables,
    two_phase_hu.Equations,
    ConstitutiveLawCase2,
    two_phase_hu.BoundaryConditionsPressureMass,
    InitialConditionCase2,
    two_phase_hu.SolutionStrategyPressureMass,
    GeometryCase2,
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
    print("t_0 = ", u_0)
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

            self.root_path = "./case_2/hu/"

            self.output_file_name = self.root_path + "OUTPUT_NEWTON_INFO"
            self.mass_output_file_name = self.root_path + "MASS_OVER_TIME"
            self.flips_file_name = self.root_path + "FLIPS"
            self.beta_file_name = self.root_path + "BETA"

    os.system("mkdir -p ./case_2/hu/")
    os.system("mkdir -p ./case_2/hu/BETA")
    folder_name = "./case_2/hu/visualization"

    time_manager = two_phase_hu.TimeManagerPP(
        schedule=np.array([0, 0.05]) / t_0,
        dt_init=2e-3 / t_0,
        dt_min_max=np.array([1e-5, 2e-3]) / t_0,
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
        "max_iterations": 15,
        "nl_convergence_tol": 1e-5,
        "nl_divergence_tol": 1e0,
        "time_manager": time_manager,
        "folder_name": folder_name,
        "meshing_kwargs": meshing_kwargs,
    }

    model = FinalModel(params)

    pp.run_time_dependent_model(model, params)
