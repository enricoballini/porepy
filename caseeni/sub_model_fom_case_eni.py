import os
import sys
import pdb
import warnings
import inspect
import copy
import time
from functools import partial
from typing import Callable, Optional, Sequence, cast


# if "/home/inspiron/Desktop/PhD/porepy/src" in sys.path:
#     sys.path.remove("/home/inspiron/Desktop/PhD/porepy/src")
#     sys.path.append("/home/inspiron/Desktop/PhD/eni_venv/porepy/src")

pp_path = "../porepy/src"
sentinel = False
for i in sys.path:
    if i == pp_path:
        sentinel = True
if not sentinel:
    sys.path.append(pp_path)
    # sys.path.append("/g100_work/pM

import numpy as np
import scipy as sp
from scipy import sparse as sps
from matplotlib import pyplot as plt
import porepy as pp
from porepy.applications.md_grids.model_geometries import (
    SquareDomainOrthogonalFractures,
)

from porepy.models import constitutive_laws
from porepy.fracs.fracture_network_3d import FractureNetwork3d

Scalar = pp.ad.Scalar

# os.system("clear")

# print("pp.__file__ = ", pp.__file__)
# print("np.__file__ = ", np.__file__)
# print("__name__ = ", __name__)


class VariablesMomentumBalance:
    """ """

    create_boundary_operator: Callable[
        [str, Sequence[pp.BoundaryGrid]], pp.ad.TimeDependentDenseArray
    ]

    def create_variables(self) -> None:
        """"""

        self.equation_system.create_variables(
            dof_info={"cells": self.nd},
            name=self.displacement_variable,
            subdomains=self.mdg.subdomains(dim=self.nd),
            tags={"si_units": "m"},
        )
        self.equation_system.create_variables(
            dof_info={"cells": self.nd},
            name=self.interface_displacement_variable,
            interfaces=self.mdg.interfaces(dim=self.nd - 1, codim=1),
            tags={"si_units": "m"},
        )
        self.equation_system.create_variables(
            dof_info={"cells": self.nd},
            name=self.contact_traction_variable,
            subdomains=self.mdg.subdomains(dim=self.nd - 1),
            tags={"si_units": "Pa"},
        )

    def displacement(self, domains: pp.SubdomainsOrBoundaries) -> pp.ad.Operator:
        """ """
        if len(domains) == 0 or all(
            isinstance(grid, pp.BoundaryGrid) for grid in domains
        ):
            return self.create_boundary_operator(  # type: ignore[call-arg]
                name=self.displacement_variable, domains=domains
            )

        if not all(isinstance(grid, pp.Grid) for grid in domains):
            raise ValueError(
                "Method called on a mixture of subdomain and boundary grids."
            )
        domains = cast(list[pp.Grid], domains)

        if not all([grid.dim == self.nd for grid in domains]):
            raise ValueError(
                "Displacement is only defined in subdomains of dimension nd."
            )

        return self.equation_system.md_variable(self.displacement_variable, domains)

    def interface_displacement(self, interfaces: list[pp.MortarGrid]) -> pp.ad.Variable:
        """ """
        if not all([intf.dim == self.nd - 1 for intf in interfaces]):
            raise ValueError(
                "Interface displacement is only defined on interfaces of dimension "
                "nd - 1."
            )

        return self.equation_system.md_variable(
            self.interface_displacement_variable, interfaces
        )

    def contact_traction(self, subdomains: list[pp.Grid]) -> pp.ad.Variable:
        """ """
        for sd in subdomains:
            if sd.dim != self.nd - 1:
                raise ValueError("Contact traction only defined on fractures")

        return self.equation_system.md_variable(
            self.contact_traction_variable, subdomains
        )

    def displacement_jump(self, subdomains: list[pp.Grid]) -> pp.ad.Operator:
        """ """
        if not all([sd.dim == self.nd - 1 for sd in subdomains]):
            raise ValueError("Displacement jump only defined on fractures")

        interfaces = self.subdomains_to_interfaces(subdomains, [1])

        interfaces = [intf for intf in interfaces if intf.dim == self.nd - 1]
        mortar_projection = pp.ad.MortarProjections(
            self.mdg, subdomains, interfaces, self.nd
        )

        rotated_jumps: pp.ad.Operator = (
            self.local_coordinates(subdomains)
            @ mortar_projection.mortar_to_secondary_avg
            @ mortar_projection.sign_of_mortar_sides
            @ self.interface_displacement(interfaces)
        )
        rotated_jumps.set_name("Rotated_displacement_jump")
        return rotated_jumps


class MomentumBalanceEquations(
    pp.BalanceEquation
):  # -----------------------------------------------------------------------------------------------------------------
    """ """

    def set_equations(self) -> None:
        """ """
        matrix_subdomains = self.mdg.subdomains(dim=self.nd)
        fracture_subdomains = self.mdg.subdomains(dim=self.nd - 1)
        interfaces = self.mdg.interfaces(dim=self.nd - 1, codim=1)

        matrix_eq = self.momentum_balance_equation(matrix_subdomains)
        fracture_eq_normal = self.normal_fracture_deformation_equation(
            fracture_subdomains
        )
        fracture_eq_tangential = self.tangential_fracture_deformation_equation(
            fracture_subdomains
        )
        intf_eq = self.interface_force_balance_equation(interfaces)

        self.equation_system.set_equation(
            matrix_eq, matrix_subdomains, {"cells": self.nd}
        )
        self.equation_system.set_equation(
            fracture_eq_normal, fracture_subdomains, {"cells": 1}
        )
        self.equation_system.set_equation(
            fracture_eq_tangential, fracture_subdomains, {"cells": self.nd - 1}
        )
        self.equation_system.set_equation(intf_eq, interfaces, {"cells": self.nd})

    def momentum_balance_equation(self, subdomains: list[pp.Grid]) -> pp.ad.Operator:
        """ """
        accumulation = self.inertia(subdomains)
        # By the convention of positive tensile stress, the balance equation is
        # acceleration - stress = body_force. The balance_equation method will *add* the
        # surface term (stress), so we need to multiply by -1.
        stress = pp.ad.Scalar(-1) * self.stress(subdomains)
        body_force = self.body_force(subdomains)
        grad_pressure = self.pressure_source(subdomains)
        source_term = body_force + grad_pressure

        equation = self.balance_equation(
            subdomains=subdomains,
            accumulation=accumulation,
            surface_term=stress,
            source=source_term,
            dim=self.nd,
        )
        equation.set_name("momentum_balance_equation")
        return equation

    def inertia(self, subdomains: list[pp.Grid]) -> pp.ad.Operator:
        """ """
        return pp.ad.Scalar(0)

    def interface_force_balance_equation(
        self,
        interfaces: list[pp.MortarGrid],
    ) -> pp.ad.Operator:
        """ """
        # Check that the interface is a fracture-matrix interface.
        for interface in interfaces:
            if interface.dim != self.nd - 1:
                raise ValueError("Interface must be a fracture-matrix interface.")

        subdomains = self.interfaces_to_subdomains(interfaces)

        matrix_subdomains = [sd for sd in subdomains if sd.dim == self.nd]

        # Geometry related
        mortar_projection = pp.ad.MortarProjections(
            self.mdg, subdomains, interfaces, self.nd
        )
        proj = pp.ad.SubdomainProjections(subdomains, self.nd)

        contact_from_primary_mortar = (
            mortar_projection.primary_to_mortar_int
            @ proj.face_prolongation(matrix_subdomains)
            @ self.internal_boundary_normal_to_outwards(
                matrix_subdomains, dim=self.nd  # type: ignore[call-arg]
            )
            @ self.stress(matrix_subdomains)
        )
        # Traction from the actual contact force.
        traction_from_secondary = self.fracture_stress(interfaces)

        force_balance_eq: pp.ad.Operator = (
            contact_from_primary_mortar
            + self.volume_integral(traction_from_secondary, interfaces, dim=self.nd)
        )
        force_balance_eq.set_name("interface_force_balance_equation")
        return force_balance_eq

    def normal_fracture_deformation_equation(
        self, subdomains: list[pp.Grid]
    ) -> pp.ad.Operator:
        """ """
        # Variables
        nd_vec_to_normal = self.normal_component(subdomains)
        # The normal component of the contact traction and the displacement jump
        t_n: pp.ad.Operator = nd_vec_to_normal @ self.contact_traction(subdomains)
        u_n: pp.ad.Operator = nd_vec_to_normal @ self.displacement_jump(subdomains)

        # Maximum function
        num_cells: int = sum([sd.num_cells for sd in subdomains])
        max_function = pp.ad.Function(pp.ad.maximum, "max_function")
        zeros_frac = pp.ad.DenseArray(np.zeros(num_cells), "zeros_frac")

        # The complimentarity condition
        equation: pp.ad.Operator = t_n + max_function(
            pp.ad.Scalar(-1.0) * t_n
            - self.contact_mechanics_numerical_constant(subdomains)
            * (u_n - self.fracture_gap(subdomains)),
            zeros_frac,
        )
        equation.set_name("normal_fracture_deformation_equation")
        return equation

    def tangential_fracture_deformation_equation(
        self,
        subdomains: list[pp.Grid],
    ) -> pp.ad.Operator:
        """ """
        num_cells = sum([sd.num_cells for sd in subdomains])

        nd_vec_to_tangential = self.tangential_component(subdomains)

        tangential_basis: list[pp.ad.SparseArray] = self.basis(
            subdomains, dim=self.nd - 1  # type: ignore[call-arg]
        )

        scalar_to_tangential = pp.ad.sum_operator_list(
            [e_i for e_i in tangential_basis]
        )

        t_t: pp.ad.Operator = nd_vec_to_tangential @ self.contact_traction(subdomains)
        u_t: pp.ad.Operator = nd_vec_to_tangential @ self.displacement_jump(subdomains)

        u_t_increment: pp.ad.Operator = pp.ad.time_increment(u_t)

        ones_frac = pp.ad.DenseArray(np.ones(num_cells * (self.nd - 1)))
        zeros_frac = pp.ad.DenseArray(np.zeros(num_cells))

        f_max = pp.ad.Function(pp.ad.maximum, "max_function")
        f_norm = pp.ad.Function(partial(pp.ad.l2_norm, self.nd - 1), "norm_function")

        tol = 1e-5  # FIXME: Revisit this tolerance!

        f_characteristic = pp.ad.Function(
            partial(pp.ad.functions.characteristic_function, tol),
            "characteristic_function_for_zero_normal_traction",
        )

        c_num_as_scalar = self.contact_mechanics_numerical_constant(subdomains)

        c_num = pp.ad.sum_operator_list(
            [e_i * c_num_as_scalar * e_i.T for e_i in tangential_basis]
        )

        tangential_sum = t_t + c_num @ u_t_increment

        norm_tangential_sum = f_norm(tangential_sum)
        norm_tangential_sum.set_name("norm_tangential")

        b_p = f_max(self.friction_bound(subdomains), zeros_frac)
        b_p.set_name("bp")

        bp_tang = (scalar_to_tangential @ b_p) * tangential_sum

        maxbp_abs = scalar_to_tangential @ f_max(b_p, norm_tangential_sum)
        characteristic: pp.ad.Operator = scalar_to_tangential @ f_characteristic(b_p)
        characteristic.set_name("characteristic_function_of_b_p")

        equation: pp.ad.Operator = (ones_frac - characteristic) * (
            bp_tang - maxbp_abs * t_t
        ) + characteristic * t_t
        equation.set_name("tangential_fracture_deformation_equation")
        return equation

    def body_force(self, subdomains: list[pp.Grid]) -> pp.ad.Operator:
        """EB: there was this comment: FIXME: See FluidMassBalanceEquations.fluid_source."""

        vals = []
        for sd in subdomains:

            acceleration = self.solid.convert_units(9.8, "m * s^-2")
            volume_force_mech = self.solid_density([sd]) * acceleration
            volume_force_fluid = self.water_density([sd]) * acceleration
            volume_force = volume_force_mech + volume_force_fluid

            data = np.stack(
                (
                    np.zeros(sd.num_cells),
                    np.zeros(sd.num_cells),
                    volume_force.evaluate(self.equation_system) * sd.cell_volumes,
                ),
                axis=1,
            )

            vals.append(data)
        return pp.ad.DenseArray(np.concatenate(vals).ravel(), "body_force")

    def pressure_source(self, subdomains) -> pp.ad.Operator:
        """ """
        if len(subdomains) > 1:
            raise NotImplementedError("inside pressure_source, only one 3D domain")
        sd = subdomains[0]
        pressure_vals = self.echelon_pressure
        fake_vals = np.zeros(sd.num_cells)
        fake_vals[self.reservoir_cell_ids] = (
            sd.cell_centers[0, self.reservoir_cell_ids] ** 2 * 2
        )
        pressure_vals = fake_vals

        discr = pp.Biot()
        data = {
            pp.PARAMETERS: {
                "mechanics": {
                    "bc": self.bc_type_mechanics(sd),
                    "fourth_order_tensor": self.stiffness_tensor(sd),
                    "biot_alpha": 1,
                    "inverter": "python",  # numba returns errors, sometimes...
                }
            },
            pp.DISCRETIZATION_MATRICES: {"mechanics": {}, "flow": {}},
        }

        discr.discretize(sd, data)
        grad_p = data[pp.DISCRETIZATION_MATRICES]["mechanics"]["grad_p"]

        grad_pressure = pp.ad.DenseArray(grad_p @ pressure_vals, "grad_pressure_source")
        div_grad_pressure = pp.ad.Divergence(subdomains, dim=self.nd) @ grad_pressure
        div_grad_pressure.set_name("div_grad_pressure")
        return div_grad_pressure


class ConstitutiveLawsMomentumBalanceEni(
    constitutive_laws.LinearElasticSolid,
    constitutive_laws.FractureGap,
    constitutive_laws.FrictionBound,
    constitutive_laws.DimensionReduction,
):  # -----------------------------------------------------------------------------------------------------------------
    def solid_density(self, subdomains: list[pp.Grid]) -> pp.ad.Scalar:
        """Constant solid density."""

        for sd in subdomains:
            if sd.dim == 3:
                vals = 2600 * np.ones(subdomains[0].num_cells)
            else:
                print("\n\n inside solid_density, there should be only one 3D grid")
        return pp.ad.DenseArray(vals, "solid_density")

    def water_density(self, subdomains: list[pp.Grid]) -> pp.ad.Scalar:
        """ """
        for sd in subdomains:
            if sd.dim == 3:
                vals = 1000 * np.ones(subdomains[0].num_cells)
            else:
                print("\n\n inside water_density, there should be only one 3D grid")
        return pp.ad.DenseArray(vals, "water_density")

    def youngs_modulus(self, subdomains: list[pp.Grid]) -> pp.ad.Operator:
        """Young's modulus [Pa]."""

        for sd in subdomains:
            if sd.dim == 3:
                E = 5.71e10 * np.ones(sd.num_cells)  # Pa
                E[self.reservoir_cell_ids] = 1e9  # *self.mu_param[0]
            else:
                print("\n\ninside young_modulus, there should be only one 3D grid")

        return pp.ad.DenseArray(E, "youngs_modulus")

    def poisson_ratio(self, subdomains: list[pp.Grid]) -> pp.ad.Operator:
        """Poisson's ration.
        EB added it. Where is it in pp?
        """

        for sd in subdomains:
            if sd.dim == 3:
                nu = self.nu * np.ones(sd.num_cells)
            else:
                print("\n\ninside poisson_ratio, there should be only one 3D grid")

        return pp.ad.DenseArray(nu, "youngs_modulus")

    def shear_modulus(self, subdomains: list[pp.Grid]) -> pp.ad.Operator:
        """Shear modulus [Pa].
        mu = G = second Lame's param = shear modulus
        """
        mu = self.youngs_modulus(subdomains) / (
            pp.ad.Scalar(2) * (pp.ad.Scalar(1) + self.poisson_ratio(subdomains))
        )
        mu.set_name("shear_modulus")
        return mu

    def lame_lambda(self, subdomains: list[pp.Grid]) -> pp.ad.Operator:
        """Lame's first parameter [Pa]."""
        lmbda = (self.youngs_modulus(subdomains) * self.poisson_ratio(subdomains)) / (
            (pp.ad.Scalar(1) + self.poisson_ratio(subdomains))
            * (pp.ad.Scalar(1) - pp.ad.Scalar(2) * self.poisson_ratio(subdomains))
        )
        lmbda.set_name("lame_lambda")
        return lmbda

    def stiffness_tensor(self, subdomain: pp.Grid) -> pp.FourthOrderTensor:
        """Stiffness tensor [Pa]."""
        lmbda = self.lame_lambda([subdomain])  # this is first Lame's param
        mu = self.shear_modulus([subdomain])  # this is second Lame's param

        # TODO: no point in working with opertors up to here
        lmbda = lmbda.evaluate(self.equation_system)
        mu = mu.evaluate(self.equation_system)

        return pp.FourthOrderTensor(mu, lmbda)

    def bulk_modulus(self, subdomains: list[pp.Grid]) -> pp.ad.Operator:
        """Bulk modulus [Pa].
        OK
        i dont use it
        """
        warnings.warning("\n\nYOU SHOULD NOT USE bulk_modulus")
        val = self.solid.lame_lambda() + 2 / 3 * self.solid.shear_modulus()
        return Scalar(val, "bulk_modulus")

    def stress(self, domains: pp.SubdomainsOrBoundaries) -> pp.ad.Operator:
        """ """
        return self.mechanical_stress(domains)


class BoundaryConditionsMomentumBalance(
    pp.BoundaryConditionMixin
):  # -------------------------------------------------------------------------------------------------------------------------------------------------

    def bc_type_mechanics(self, sd: pp.Grid) -> pp.BoundaryConditionVectorial:
        """ """
        boundary_faces = self.domain_boundary_sides(sd).all_bf
        bc = pp.BoundaryConditionVectorial(sd, boundary_faces, "dir")
        bc.internal_to_dirichlet(sd)
        return bc

    def bc_values_displacement(self, boundary_grid: pp.BoundaryGrid) -> np.ndarray:
        """
        this is what you need to modify to change bc part 1 of 2
        """
        values = np.zeros((self.nd, boundary_grid.num_cells))
        bounds = self.domain_boundary_sides(boundary_grid)

        values[0][bounds.west] += self.solid.convert_units(0, "m")
        # values[1][bounds.west] += self.solid.convert_units(0, "m")
        # values[2][bounds.west] += self.solid.convert_units(0, "m")

        values[0][bounds.east] += self.solid.convert_units(0, "m")
        # values[1][bounds.east] += self.solid.convert_units(0, "m")
        # values[2][bounds.east] += self.solid.convert_units(0, "m")

        # values[0][bounds.north] += self.solid.convert_units(0, "m")
        values[1][bounds.north] += self.solid.convert_units(0, "m")
        # values[2][bounds.north] += self.solid.convert_units(0, "m")

        # values[0][bounds.south] += self.solid.convert_units(0, "m")
        values[1][bounds.south] += self.solid.convert_units(0, "m")
        # values[2][bounds.south] += self.solid.convert_units(0, "m")

        # values[0][bounds.bottom] += self.solid.convert_units(0, "m")
        # values[1][bounds.bottom] += self.solid.convert_units(0, "m")
        values[2][bounds.bottom] += self.solid.convert_units(0, "m")

        return values.ravel("F")

    def bc_values_stress(self, boundary_grid: pp.BoundaryGrid) -> np.ndarray:
        """
        this is what you need to modify to change bc part 2 of 2
        """

        values = 0 * np.ones((self.nd, boundary_grid.num_cells))
        bounds = self.domain_boundary_sides(boundary_grid)

        values[0][bounds.top] *= self.solid.convert_units(0, "Pa")
        values[1][bounds.top] *= self.solid.convert_units(0, "Pa")
        values[2][bounds.top] *= self.solid.convert_units(0, "Pa")

        return values.ravel("F")

    def update_all_boundary_conditions(self) -> None:
        """Set values for the displacement and the stress on boundaries."""
        super().update_all_boundary_conditions()
        self.update_boundary_condition(
            self.displacement_variable, self.bc_values_displacement
        )
        self.update_boundary_condition(self.stress_keyword, self.bc_values_stress)


class GeometryCloseToEni(
    pp.ModelGeometry
):  # ----------------------------------------------------------------------------------------------------------------------------------
    def meshing_arguments(self) -> dict[str, float]:
        """ """
        cell_size = self.solid.convert_units(0.05, "m")
        default_meshing_args: dict[str, float] = {"cell_size": cell_size}
        return self.params.get("meshing_arguments", default_meshing_args)

    def set_geometry(self) -> None:
        """ """
        self.set_domain()
        eni_grid = self.load_eni_grid(path_to_mat="../egridtoporepy/mrst_grid")

        self.xmin = -1000
        self.xmax = 3000
        self.ymin = -500
        self.ymax = 1500
        width = 550  # 1625  # 525  # 125  # step 125
        # self.ymax = self.ymin + width
        self.zmin = 0
        self.zmax = 2500

        self.reservoir_z_left_top = 1450  # from paper
        self.reservoir_z_left_bottom = 1550
        self.reservoir_x_west = 0
        self.reservoir_x_east = 2000
        self.reservoir_y_south = 0
        self.reservoir_y_north = 1000

        self.reservoir_z_right_top = 1450 + 50 * np.sin(80 * np.pi / 180)  # from paper
        self.reservoir_z_right_bottom = 1550 + 50 * np.sin(80 * np.pi / 180)

        # ind_cut = (
        #     eni_grid.cell_centers[1, :] < self.ymin + width
        # )  # old_grid: + 2000 => 24000 cell, more or less the limit for my computer
        # [_, eni_grid], _, _ = pp.partition.partition_grid(eni_grid, ind_cut)

        polygon_vertices = np.array(
            [
                [
                    535.51,
                    535.51,
                    976.327,
                    976.327,
                ],  # x
                [
                    -500,
                    1500,
                    1500,
                    -500,
                ],  # y
                [
                    0,
                    0,
                    2500,
                    2500,
                ],  # z
            ]
        )

        self.fracture_faces_id = self.find_fracture_faces(eni_grid, polygon_vertices)
        # self.plot_fracture_nodes(eni_grid)
        self.create_frac_sd_for_plot(eni_grid, self.fracture_faces_id)

        self.reservoir_cell_ids = self.find_reservoir_cells(eni_grid, polygon_vertices)

        self.mdg = pp.MixedDimensionalGrid()
        self.mdg.add_subdomains(eni_grid)
        self.mdg.compute_geometry()
        self.mdg.set_boundary_grid_projections()  # I added it. Where should it be called normally?

        self.nd: int = self.mdg.dim_max()

        pp.set_local_coordinate_projections(self.mdg)

        self.set_well_network()
        if len(self.well_network.wells) > 0:
            # Compute intersections
            assert isinstance(self.fracture_network, FractureNetwork3d)
            pp.compute_well_fracture_intersections(
                self.well_network, self.fracture_network
            )
            self.well_network.mesh(self.mdg)

    def set_domain(self) -> None:
        """ """
        box = {
            "xmin": 0,
            "xmax": 1,
            "ymin": 0,
            "ymax": 1,
            "zmin": 0,
            "zmax": 1,
        }  # TODO: remove it
        self._domain = pp.domain.Domain(bounding_box=box)

    def set_fractures(self) -> None:
        """ """
        # [[x_min, x_max],[y_min, y_max]]
        # frac_1 = pp.LineFracture(np.array([[3, 3], [0, 0.5]]))

        # [[pt1], [pt2], [], []].T
        # frac_1 = pp.PlaneFracture(
        #     np.array([[3, 0, 0], [3, 1, 0], [3, 1, 0.5], [3, 0, 0.5]]).T
        # )
        self._fractures = []

    def load_eni_grid(self, path_to_mat):
        """
        see files in read_ecl_grid
        """
        mrst_grid = sp.io.loadmat(path_to_mat)

        nodes = mrst_grid["node_coord"].T.astype(np.float64)
        fn_row = mrst_grid["fn_node_ind"].astype(np.int32).ravel() - 1
        fn_data = np.ones(fn_row.size, dtype=bool)

        indptr = mrst_grid["fn_indptr"].astype(np.int32).ravel() - 1
        fn = sps.csc_matrix(
            (fn_data, fn_row, indptr),
            shape=(
                fn_row.max() + 1,
                indptr.shape[0] - 1,
            ),  # indptr.shape[0] - 1 a caso...
        )

        cf_row = mrst_grid["cf_face_ind"].astype(np.int32).ravel() - 1
        cf_col = mrst_grid["cf_cell_ind"].astype(np.int32).ravel() - 1
        cf_data = mrst_grid["cf_sgn"].ravel().astype(np.float64)

        cf = sps.coo_matrix(
            (cf_data, (cf_row, cf_col)), shape=(cf_row.max() + 1, cf_col.max() + 1)
        ).tocsc()

        dim = nodes.shape[0]

        g = pp.Grid(dim, nodes, fn, cf, "myname")
        g.compute_geometry()

        return g

    def find_fracture_faces(self, sd, vertices_polygon):
        """ """
        points = sd.face_centers
        distances, _, _ = pp.distances.points_polygon(
            points, vertices_polygon, tol=1e-5
        )
        faces_on_frac_id = np.where(np.isclose(0, distances, rtol=0, atol=1e-1))[
            0
        ]  # pay attention, the tolerance is high...
        return faces_on_frac_id

    def find_reservoir_cells(self, sd, vertices_polygon):
        """ """
        points = sd.cell_centers
        distances, _, _ = pp.distances.points_polygon_signed(
            points, vertices_polygon, tol=1e-5
        )
        left_ids = np.where(distances < 0)[0]
        right_ids = np.where(distances >= 0)[0]

        # z bounds:
        # pay attention, you work with indices of a subset
        reservoir_ids_z_bnd_left = np.arange(sd.num_cells)[left_ids][
            np.where(
                np.logical_and(
                    points[2][left_ids] > self.reservoir_z_left_top,
                    points[2][left_ids] < self.reservoir_z_left_bottom,
                )
            )[0]
        ]

        reservoir_ids_z_bnd_right = np.arange(sd.num_cells)[right_ids][
            np.where(
                np.logical_and(
                    points[2][right_ids] > self.reservoir_z_right_top,
                    points[2][right_ids] < self.reservoir_z_right_bottom,
                )
            )[0]
        ]

        # x bounds:
        reservoir_ids_x_bnd = np.where(
            np.logical_and(
                points[0] > self.reservoir_x_west,
                points[0] < self.reservoir_x_east,
            )
        )[0]

        # y bounds:
        reservoir_ids_y_bnd = np.where(
            np.logical_and(
                points[1] > self.reservoir_y_south,
                points[1] < self.reservoir_y_north,
            )
        )[0]

        reservoir_ids_z_bnd = np.concatenate(
            (reservoir_ids_z_bnd_left, reservoir_ids_z_bnd_right)
        )
        tmp = np.intersect1d(reservoir_ids_z_bnd, reservoir_ids_x_bnd)
        self.reservoir_cell_ids = np.intersect1d(tmp, reservoir_ids_y_bnd)

        subgrid, _, _ = pp.partition.extract_subgrid(
            sd, self.reservoir_cell_ids, faces=False
        )
        exporter = pp.Exporter(
            subgrid, "reservoir" + self.subscript, folder_name=self.save_folder
        )
        exporter.write_vtu()

    def create_frac_sd_for_plot(self, sd, faces_fract_id):
        """ """
        self.sd_fract, _, _ = pp.partition.extract_subgrid(
            sd, faces_fract_id, faces=True
        )

    def plot_fracture_nodes(self, sd):
        """ """
        nodes_id = sd.face_nodes @ sp.sparse.coo_array(
            (
                [True] * self.fracture_faces_id.shape[0],
                (self.fracture_faces_id, np.zeros(self.fracture_faces_id.shape[0])),
            ),
            shape=(sd.face_nodes.shape[1], 1),
        )
        nodes = sd.nodes[:, np.ravel(nodes_id.todense())]
        fig = plt.figure()
        ax = fig.add_subplot(projection="3d")
        ax.scatter(nodes[0], nodes[1], nodes[2], marker=".")
        plt.show()

    def domain_boundary_sides(
        self, domain: pp.Grid | pp.BoundaryGrid, tol: Optional[float] = 1e-10
    ) -> pp.domain.DomainSides:
        """
        # z pointing downwards

        # west -> towards negative x
        # east -> towards positive x
        # south -> neg y
        # north -> pos y
        # top -> neg z
        # bottom -> positive z
        """
        if isinstance(domain, pp.Grid):
            face_centers = domain.face_centers
            num_faces = domain.num_faces
            all_bf = domain.get_boundary_faces()
        elif isinstance(domain, pp.BoundaryGrid):
            face_centers = domain.cell_centers
            num_faces = domain.num_cells
            all_bf = np.arange(num_faces)
        else:
            raise ValueError(
                "Domain must be either Grid or BoundaryGrid. Provided:", domain
            )

        east = np.abs(self.xmax - face_centers[0]) <= tol
        west = np.abs(self.xmin - face_centers[0]) <= tol
        if self.mdg.dim_max() == 1:
            north = np.zeros(num_faces, dtype=bool)
            south = north.copy()
        else:
            north = np.abs(self.ymax - face_centers[1]) <= tol
            south = np.abs(self.ymin - face_centers[1]) <= tol
        if self.mdg.dim_max() < 3:
            top = np.zeros(num_faces, dtype=bool)
            bottom = top.copy()
        else:
            top = face_centers[2] > self.zmax - tol
            bottom = face_centers[2] < self.zmin + tol

        domain_sides = pp.domain.DomainSides(
            all_bf, east, west, north, south, top, bottom
        )

        # fig = plt.figure()
        # ax = fig.add_subplot(projection="3d")
        # ax.scatter(
        #     domain.cell_centers[0],
        #     domain.cell_centers[1],
        #     domain.cell_centers[2],
        #     marker=".",
        # )
        # plt.show()

        # fig = plt.figure()
        # ax = fig.add_subplot(projection="3d")
        # ax.scatter(
        #     domain.cell_centers[0][west],
        #     domain.cell_centers[1][west],
        #     domain.cell_centers[2][west],
        #     marker=".",
        # )
        # plt.title("west")
        # plt.show()

        # fig = plt.figure()
        # ax = fig.add_subplot(projection="3d")
        # ax.scatter(
        #     domain.cell_centers[0][east],
        #     domain.cell_centers[1][east],
        #     domain.cell_centers[2][east],
        #     marker=".",
        # )
        # plt.title("east")
        # plt.show()

        # fig = plt.figure()
        # ax = fig.add_subplot(projection="3d")
        # ax.scatter(
        #     domain.cell_centers[0][south],
        #     domain.cell_centers[1][south],
        #     domain.cell_centers[2][south],
        #     marker=".",
        # )
        # plt.title("south")
        # plt.show()

        # fig = plt.figure()
        # ax = fig.add_subplot(projection="3d")
        # ax.scatter(
        #     domain.cell_centers[0][north],
        #     domain.cell_centers[1][north],
        #     domain.cell_centers[2][north],
        #     marker=".",
        # )
        # plt.title("north")
        # plt.show()

        # fig = plt.figure()
        # ax = fig.add_subplot(projection="3d")
        # ax.scatter(
        #     domain.cell_centers[0][bottom],
        #     domain.cell_centers[1][bottom],
        #     domain.cell_centers[2][bottom],
        #     marker=".",
        # )
        # plt.title("bottom")
        # plt.show()

        # fig = plt.figure()
        # ax = fig.add_subplot(projection="3d")
        # ax.scatter(
        #     domain.cell_centers[0][top],
        #     domain.cell_centers[1][top],
        #     domain.cell_centers[2][top],
        #     marker=".",
        # )
        # plt.title("top")
        # plt.show()

        return domain_sides


class SolutionStrategyMomentumBalance(
    pp.SolutionStrategy
):  # ------------------------------------------------------------------------------------------------------------------
    """ """

    def __init__(self, params: Optional[dict] = None) -> None:
        super().__init__(params)

        self.displacement_variable: str = "u"
        self.interface_displacement_variable: str = "u_interface"
        self.contact_traction_variable: str = "t"
        self.stress_keyword: str = "mechanics"

    def prepare_simulation(self) -> None:
        """ """

        t1 = time.time()

        self.clean_working_directory()

        self.set_materials()
        self.set_geometry()

        self.initialize_data_saving(
            exporter_folder=self.exporter_folder, subscript=self.subscript
        )
        self.set_equation_system_manager()
        self.create_variables()
        self.initial_condition()
        self.reset_state_from_file()

        self.set_equations()
        self.set_discretization_parameters()
        self.discretize()
        self._initialize_linear_solver()
        self.set_nonlinear_discretizations()

        self.save_data_time_step()

        print("\n\n\n ONE RUN TIME = ", time.time() - t1)
        print("\n\n\n")

    def clean_working_directory(self):
        """ """
        os.system("rm *.pvd *.vtu")
        os.system("rm -r visualization")

    def initial_condition(self) -> None:
        """ """
        super().initial_condition()

        num_frac_cells = sum(
            sd.num_cells for sd in self.mdg.subdomains(dim=self.nd - 1)
        )
        traction_vals = np.zeros((self.nd, num_frac_cells))
        traction_vals[-1] = self.solid.convert_units(-1, "Pa")
        self.equation_system.set_variable_values(
            traction_vals.ravel("F"),
            [self.contact_traction_variable],
            time_step_index=0,
            iterate_index=0,
        )

    def set_discretization_parameters(self) -> None:
        """ """

        super().set_discretization_parameters()
        for sd, data in self.mdg.subdomains(return_data=True):
            if sd.dim == self.nd:
                pp.initialize_data(
                    sd,
                    data,
                    self.stress_keyword,
                    {
                        "bc": self.bc_type_mechanics(sd),
                        "fourth_order_tensor": self.stiffness_tensor(sd),
                    },
                )
                data[pp.PARAMETERS]["mechanics"][
                    "inverter"
                ] = "python"  # see pp.matrix_operations.invert_diagonal_blocks

    def contact_mechanics_numerical_constant(
        self, subdomains: list[pp.Grid]
    ) -> pp.ad.Scalar:
        """ """
        shear_modulus = self.solid.shear_modulus()
        characteristic_distance = (
            self.solid.residual_aperture() + self.solid.fracture_gap()
        )
        softening_factor = 1e-1
        val = softening_factor * shear_modulus / characteristic_distance
        return pp.ad.Scalar(val, name="Contact_mechanics_numerical_constant")

    def _is_nonlinear_problem(self) -> bool:
        """ """
        return self.mdg.dim_min() < self.nd

    def after_simulation(self) -> None:
        """ """
        # no way, the lynear system is too big
        # A = self.linear_system[0]
        # eigvals, _ = sp.sparse.linalg.eigs(A, k=int(A.shape[0] / 100))
        # print(max(eigvals))
        # print(min(eigvals))
        # _, sss, _ = sp.sparse.linalg.svds(A, k=int(A.shape[0] / 1000))
        # cond = np.linalg.cond(A.todense(), p=2)
        # print("condition number K2 = ", cond)

        subdomains_data = self.mdg.subdomains(return_data=True)
        subdomains = [subdomains_data[0][0]]
        sd = subdomains[0]
        data = subdomains_data[0][1]
        boundary_grids = self.mdg.boundaries(return_data=False)

        stress_tensor_grad = data[pp.DISCRETIZATION_MATRICES][self.stress_keyword][
            "stress"
        ]
        bound_stress = data[pp.DISCRETIZATION_MATRICES][self.stress_keyword][
            "bound_stress"
        ]  # not 100% clear what is this tensor. It multiplies u_b that contains both dir and neumann

        u = self.displacement(subdomains).evaluate(self.equation_system).val
        # same of np.linalg.solve(self.linear_system[0], self.linear_system[1]) => no reshape problems

        u_b_displ = self.bc_values_displacement(boundary_grids[0])
        u_b_stress = self.bc_values_stress(boundary_grids[0])

        u_b = (
            u_b_displ + u_b_stress
        )  # they should be exclusive, either displ or stress is != 0

        u_b_filled = np.zeros((self.nd, sd.num_faces))
        u_b_filled[:, sd.get_all_boundary_faces()] = u_b.reshape((3, -1), order="F")
        u_b_filled = u_b_filled.ravel("F")
        T = stress_tensor_grad * u + bound_stress * u_b_filled

        T_vect = np.reshape(T, (sd.dim, -1), order="F")
        T_vect_frac = T_vect[:, self.fracture_faces_id]

        # pp.plot_grid(sd, vector_value=T_vect, figsize=(15, 12), alpha=0)
        # pp.plot_grid(
        #     self.sd_fract, T_vect_frac, alpha=0
        # )  # NO, for pp self.sd_fract is 2D, T_vect_frac is 3D, so they don't match, see below
        # T_vect_frac_filled = np.zeros((self.nd, sd.num_faces))
        # T_vect_frac_filled[:, self.fracture_faces_id] = T_vect_frac
        # pp.plot_grid(sd, vector_value=10000 * T_vect_frac_filled, alpha=0) # there is an eror in paraview... don't trust it

        exporter = pp.Exporter(
            self.sd_fract,
            file_name="sd_fract" + self.subscript,
            folder_name=self.save_folder,
        )
        exporter.write_vtu(
            [
                (self.sd_fract, "T_x", T_vect_frac[0]),
                (self.sd_fract, "T_y", T_vect_frac[1]),
                (self.sd_fract, "T_z", T_vect_frac[2]),
            ]
        )

        normal = sd.face_normals[:, self.fracture_faces_id][
            :, 0
        ]  # the fracture is planar, i take the first vecor as ref
        normal = normal / np.linalg.norm(normal, ord=2)
        normal_projection = pp.map_geometry.normal_matrix(normal=normal)
        tangential_projetion = pp.map_geometry.tangent_matrix(normal=normal)

        T_normal = normal_projection @ T_vect_frac
        T_normal_norm = np.linalg.norm(T_normal.T, ord=2, axis=1)
        T_tangential = tangential_projetion @ T_vect_frac
        T_tangential_norm = np.linalg.norm(T_tangential.T, ord=2, axis=1)

        # assert np.all(
        #     np.isclose(
        #         np.sqrt(T_tangential_norm**2 + T_normal_norm**2),
        #         np.linalg.norm(T_vect_frac.T, axis=1),
        #         rtol=0,
        #         atol=1e-8,
        #     )
        # )

        # stress tensor (sigma) computation: ------------------------------------------------
        # sigma_tensor = self.stress(subdomains).evaluate(self.equation_system).val  # NO!
        # is it possible?

        # # useless export: --------------------------------------
        # exporter = pp.Exporter(self.mdg, file_name="eni_case", folder_name="./")
        # exporter.write_vtu("u")


class SubModelCaseEni(
    MomentumBalanceEquations,
    VariablesMomentumBalance,
    ConstitutiveLawsMomentumBalanceEni,
    BoundaryConditionsMomentumBalance,
    GeometryCloseToEni,
    SolutionStrategyMomentumBalance,
    pp.DataSavingMixin,
):
    pass


if __name__ == "__main__":
    model = SubModelCaseEni()
    model.mu_param = None
    model.echelon_pressure = None
    model.save_folder = "./"
    model.exporter_folder = "./"
    model.subscript = "_00.00"
    model.nu = 0.5

    # pp.run_time_dependent_model(model, {}) # same output of run_stationary....
    pp.run_stationary_model(model, {})
    print("\nDone!")
