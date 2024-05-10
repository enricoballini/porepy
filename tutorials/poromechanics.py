import os
import sys
import pdb
import inspect
import copy
from functools import partial
from typing import Callable, Optional, Sequence, cast

sys.path.remove("/home/inspiron/Desktop/PhD/porepy/src")
# sys.path.remove("/usr/lib/python310.zip") # I'd like to remove all the paths outside eni_env, but I need them. Not clear why
# sys.path.remove("/usr/lib/python3.10")
# sys.path.remove("/usr/lib/python3.10/lib-dynload")
sys.path.append("/home/inspiron/Desktop/PhD/eni_venv/porepy/src")

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


os.system("clear")

# print("pp.__file__ = ", pp.__file__)
# print("np.__file__ = ", np.__file__)


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
        # Check that the subdomains are grids
        if not all(isinstance(grid, pp.Grid) for grid in domains):
            raise ValueError(
                "Method called on a mixture of subdomain and boundary grids."
            )
        # Now we can cast to Grid
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
        # Check that the subdomains are fractures
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
        # Only use matrix-fracture interfaces
        interfaces = [intf for intf in interfaces if intf.dim == self.nd - 1]
        mortar_projection = pp.ad.MortarProjections(
            self.mdg, subdomains, interfaces, self.nd
        )
        # The displacement jmup is expressed in the local coordinates of the fracture.
        # First use the sign of the mortar sides to get a difference, then map first
        # from the interface to the fracture, and finally to the local coordinates.
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
        equation = self.balance_equation(
            subdomains, accumulation, stress, body_force, dim=self.nd
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
        # Split into matrix and fractures. Sort on dimension to allow for multiple
        # matrix domains. Otherwise, we could have picked the first element.
        matrix_subdomains = [sd for sd in subdomains if sd.dim == self.nd]

        # Geometry related
        mortar_projection = pp.ad.MortarProjections(
            self.mdg, subdomains, interfaces, self.nd
        )
        proj = pp.ad.SubdomainProjections(subdomains, self.nd)

        # Contact traction from primary grid and mortar displacements (via primary grid).
        # Spelled out for clarity:
        #   1) The sign of the stress on the matrix subdomain is corrected so that all
        #      stress components point outwards from the matrix (or inwards, EK is not
        #      completely sure, but the point is the consistency).
        #   2) The stress is prolonged from the matrix subdomains to all subdomains seen
        #      by the mortar grid (that is, the matrix and the fracture).
        #   3) The stress is projected to the mortar grid.
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
        # The force balance equation. Note that the force from the fracture is a
        # traction, not a stress, and must be scaled with the area of the interface.
        # This is not the case for the force from the matrix, which is a stress.
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
        # The lines below is an implementation of equations (24) and (26) in the paper
        #
        # Berge et al. (2020): Finite volume discretization for poroelastic media with
        #   fractures modeled by contact mechanics (IJNME, DOI: 10.1002/nme.6238). The
        #
        # Note that:
        #  - We do not directly implement the matrix elements of the contact traction
        #    as are derived by Berge in their equations (28)-(32). Instead, we directly
        #    implement the complimentarity function, and let the AD framework take care
        #    of the derivatives.
        #  - Related to the previous point, we do not implement the regularization that
        #    is discussed in Section 3.2.1 of the paper.

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
        # The lines below is an implementation of equations (25) and (27) in the paper
        #
        # Berge et al. (2020): Finite volume discretization for poroelastic media with
        #   fractures modeled by contact mechanics (IJNME, DOI: 10.1002/nme.6238). The
        #
        # Note that:
        #  - We do not directly implement the matrix elements of the contact traction
        #    as are derived by Berge in their equations (28)-(32). Instead, we directly
        #    implement the complimentarity function, and let the AD framework take care
        #    of the derivatives.
        #  - Related to the previous point, we do not implement the regularization that
        #    is discussed in Section 3.2.1 of the paper.

        # Basis vector combinations
        num_cells = sum([sd.num_cells for sd in subdomains])
        # Mapping from a full vector to the tangential component
        nd_vec_to_tangential = self.tangential_component(subdomains)

        # Basis vectors for the tangential components. This is a list of Ad matrices,
        # each of which represents a cell-wise basis vector which is non-zero in one
        # dimension (and this is known to be in the tangential plane of the subdomains).
        # Ignore mypy complaint on unknown keyword argument
        tangential_basis: list[pp.ad.SparseArray] = self.basis(
            subdomains, dim=self.nd - 1  # type: ignore[call-arg]
        )

        # To map a scalar to the tangential plane, we need to sum the basis vectors. The
        # individual basis functions have shape (Nc * (self.nd - 1), Nc), where Nc is
        # the total number of cells in the subdomain. The sum will have the same shape,
        # but the row corresponding to each cell will be non-zero in all rows
        # corresponding to the tangential basis vectors of this cell. EK: mypy insists
        # that the argument to sum should be a list of booleans. Ignore this error.
        scalar_to_tangential = pp.ad.sum_operator_list(
            [e_i for e_i in tangential_basis]
        )

        # Variables: The tangential component of the contact traction and the
        # displacement jump
        t_t: pp.ad.Operator = nd_vec_to_tangential @ self.contact_traction(subdomains)
        u_t: pp.ad.Operator = nd_vec_to_tangential @ self.displacement_jump(subdomains)
        # The time increment of the tangential displacement jump
        u_t_increment: pp.ad.Operator = pp.ad.time_increment(u_t)

        # Vectors needed to express the governing equations
        ones_frac = pp.ad.DenseArray(np.ones(num_cells * (self.nd - 1)))
        zeros_frac = pp.ad.DenseArray(np.zeros(num_cells))

        # Functions EK: Should we try to agree on a name convention for ad functions?
        # EK: Yes. Suggestions?
        f_max = pp.ad.Function(pp.ad.maximum, "max_function")
        f_norm = pp.ad.Function(partial(pp.ad.l2_norm, self.nd - 1), "norm_function")

        # With the active set method, the performance of the Newton solver is sensitive
        # to changes in state between sticking and sliding. To reduce the sensitivity to
        # round-off errors, we use a tolerance to allow for slight inaccuracies before
        # switching between the two cases.
        tol = 1e-5  # FIXME: Revisit this tolerance!
        # The characteristic function will evaluate to 1 if the argument is less than
        # the tolerance, and 0 otherwise.
        f_characteristic = pp.ad.Function(
            partial(pp.ad.functions.characteristic_function, tol),
            "characteristic_function_for_zero_normal_traction",
        )

        # The numerical constant is used to loosen the sensitivity in the transition
        # between sticking and sliding.
        # Expanding using only left multiplication to with scalar_to_tangential does not
        # work for an array, unlike the operators below. Arrays need right
        # multiplication as well.
        c_num_as_scalar = self.contact_mechanics_numerical_constant(subdomains)

        # The numerical parameter is a cell-wise scalar which must be extended to a
        # vector quantity to be used in the equation (multiplied from the right).
        # Spelled out, from the right: Restrict the vector quantity to one dimension in
        # the tangential plane (e_i.T), multiply with the numerical parameter, prolong
        # to the full vector quantity (e_i), and sum over all all directions in the
        # tangential plane. EK: mypy insists that the argument to sum should be a list
        # of booleans. Ignore this error.
        c_num = pp.ad.sum_operator_list(
            [e_i * c_num_as_scalar * e_i.T for e_i in tangential_basis]
        )

        # Combine the above into expressions that enter the equation. c_num will
        # effectively be a sum of SparseArrays, thus we use a matrix-vector product @
        tangential_sum = t_t + c_num @ u_t_increment

        norm_tangential_sum = f_norm(tangential_sum)
        norm_tangential_sum.set_name("norm_tangential")

        b_p = f_max(self.friction_bound(subdomains), zeros_frac)
        b_p.set_name("bp")

        # Remove parentheses to make the equation more readable if possible. The product
        # between (the SparseArray) scalar_to_tangential and b_p is of matrix-vector
        # type (thus @), and the result is then multiplied elementwise with
        # tangential_sum.
        bp_tang = (scalar_to_tangential @ b_p) * tangential_sum

        # For the use of @, see previous comment.
        maxbp_abs = scalar_to_tangential @ f_max(b_p, norm_tangential_sum)
        characteristic: pp.ad.Operator = scalar_to_tangential @ f_characteristic(b_p)
        characteristic.set_name("characteristic_function_of_b_p")

        # Compose the equation itself. The last term handles the case bound=0, in which
        # case t_t = 0 cannot be deduced from the standard version of the complementary
        # function (i.e. without the characteristic function). Filter out the other
        # terms in this case to improve convergence
        equation: pp.ad.Operator = (ones_frac - characteristic) * (
            bp_tang - maxbp_abs * t_t
        ) + characteristic * t_t
        equation.set_name("tangential_fracture_deformation_equation")
        return equation

    def body_force(self, subdomains: list[pp.Grid]) -> pp.ad.Operator:
        """EB: there was this comment: FIXME: See FluidMassBalanceEquations.fluid_source."""

        units = self.units
        vals = []
        for sd in subdomains:
            # data = np.zeros((sd.num_cells, self.nd))
            # data = np.ones((sd.num_cells, self.nd))
            data = np.stack(
                (
                    0 * np.ones(sd.num_cells),
                    0 * np.ones(sd.num_cells),
                    1 * np.ones(sd.num_cells),
                ),
                axis=1,
            )

            # if sd.dim == 2:
            #     cell_centers = sd.cell_centers
            #     indices = (
            #         (cell_centers[0] > (0.3 / units.m))
            #         & (cell_centers[0] < (0.7 / units.m))
            #         & (cell_centers[1] > (0.3 / units.m))
            #         & (cell_centers[1] < (0.7 / units.m))
            #     )

            #     acceleration = self.solid.convert_units(-9.8, "m * s^-2")
            #     force = self.solid.density() * acceleration
            #     data[indices, 1] = force * sd.cell_volumes[indices]

            vals.append(data)
        return pp.ad.DenseArray(np.concatenate(vals).ravel(), "body_force")


class ConstitutiveLawsMomentumBalance(
    constitutive_laws.LinearElasticSolid,
    constitutive_laws.FractureGap,
    constitutive_laws.FrictionBound,
    constitutive_laws.DimensionReduction,
):  # -----------------------------------------------------------------------------------------------------------------

    def stress(self, domains: pp.SubdomainsOrBoundaries) -> pp.ad.Operator:
        """ """
        # Method from constitutive library's LinearElasticRock.
        return self.mechanical_stress(domains)


class BoundaryConditionsMomentumBalance(
    pp.BoundaryConditionMixin
):  # -------------------------------------------------------------------------------------------------------------------------------------------------

    def bc_type_mechanics(self, sd: pp.Grid) -> pp.BoundaryConditionVectorial:
        """ """
        # Define boundary faces.
        boundary_faces = self.domain_boundary_sides(sd).all_bf
        bc = pp.BoundaryConditionVectorial(sd, boundary_faces, "dir")
        # Default internal BC is Neumann. We change to Dirichlet for the contact
        # problem. I.e., the mortar variable represents the displacement on the fracture
        # faces.
        bc.internal_to_dirichlet(sd)
        return bc

    def bc_values_displacement(self, boundary_grid: pp.BoundaryGrid) -> np.ndarray:
        """
        this is what you need to modify to change bc part 1 of 2
        """
        values = np.zeros((self.nd, boundary_grid.num_cells))
        bounds = self.domain_boundary_sides(boundary_grid)

        values[0][bounds.west] += self.solid.convert_units(0, "m")
        values[1][bounds.west] += self.solid.convert_units(0, "m")
        values[2][bounds.west] += self.solid.convert_units(0, "m")

        values[0][bounds.east] += self.solid.convert_units(0, "m")
        values[1][bounds.east] += self.solid.convert_units(0, "m")
        values[2][bounds.east] += self.solid.convert_units(0, "m")

        values[0][bounds.north] += self.solid.convert_units(0, "m")
        values[1][bounds.north] += self.solid.convert_units(0, "m")
        values[2][bounds.north] += self.solid.convert_units(0, "m")

        values[0][bounds.south] += self.solid.convert_units(0, "m")
        values[1][bounds.south] += self.solid.convert_units(0, "m")
        values[2][bounds.south] += self.solid.convert_units(0, "m")

        values[0][bounds.bottom] += self.solid.convert_units(0, "m")
        values[1][bounds.bottom] += self.solid.convert_units(0, "m")
        values[2][bounds.bottom] += self.solid.convert_units(0, "m")

        return values.ravel("F")
        # return np.zeros((self.nd, boundary_grid.num_cells)).ravel("F")

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
        # return np.zeros((self.nd, boundary_grid.num_cells)).ravel("F")

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
        # If meshing arguments are provided in the params, they should already be
        # scaled by the length unit.
        return self.params.get("meshing_arguments", default_meshing_args)

    def set_geometry(self) -> None:
        """ """
        # cell_size_x = self.solid.convert_units(0.2, "m")
        # cell_size_y = self.solid.convert_units(0.2, "m")
        # cell_size_z = self.solid.convert_units(0.02, "m")
        # meshing_params = {
        #     "cell_size_x": cell_size_x,
        #     "cell_size_y": cell_size_y,
        #     "cell_size_z": cell_size_z,
        # }

        self.set_domain()
        # self.set_fractures()
        # self.fracture_network = pp.create_fracture_network(self.fractures, self.domain)
        # self.mdg = pp.create_mdg(
        #     grid_type="cartesian",
        #     meshing_args=meshing_params,
        #     fracture_network=self.fracture_network,
        #     **self.meshing_kwargs(),
        # )

        eni_grid = self.load_eni_grid(
            path_to_mat="/home/inspiron/Desktop/PhD/eni_venv/egridtoporepy/mrst_grid"
        )

        # old grid:
        # self.xmin = 0
        # self.xmax = 12000
        # self.ymin = 1500
        # # self.ymax = 7000
        # width = 1000  # step 250
        # self.ymax = self.ymin + width
        # self.zmin = 2050
        # self.zmax = 2650

        self.xmin = -1000
        self.xmax = 3000
        self.ymin = -500
        # self.ymax = 1500
        width = 125  # 525  # old_grid: step 250 # new_grid step 125
        self.ymax = self.ymin + width
        self.zmin = 0
        self.zmax = 2500

        ind_cut = (
            eni_grid.cell_centers[1, :] < self.ymin + width
        )  # old_grid: + 2000 => 24000 cell, more or less the limit for my computer
        [_, eni_grid], _, _ = pp.partition.partition_grid(eni_grid, ind_cut)

        # old grid:
        # polygon_vertices = np.array(
        #     [
        #         [  # x
        #             4874.16,
        #             4874.16,
        #             5913.4,
        #             5913.4,
        #         ],
        #         [  # y
        #             1500,
        #             3500,
        #             3500,
        #             1500,
        #         ],
        #         [  # z
        #             2650,
        #             2650,
        #             2050,
        #             2050,
        #         ],
        #     ]
        # )

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

        # exporter = pp.Exporter(eni_grid, "eni_grid_cut")
        # exporter.write_vtu()

        self.mdg = pp.MixedDimensionalGrid()
        self.mdg.add_subdomains(eni_grid)
        self.mdg.compute_geometry()
        self.mdg.set_boundary_grid_projections()  # I added it. Where should it called normally?

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
        # print("\n faces_on_frac_id.shape = ", faces_on_frac_id.shape)
        return faces_on_frac_id

    def create_frac_sd_for_plot(self, sd, faces_fract_id):
        """ """
        self.sd_fract, _, _ = pp.partition.extract_subgrid(
            sd, faces_fract_id, faces=True
        )
        # pp.plot_grid(self.sd_fract, alpha=0)

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
            # print(domain)
            # non_horizontal_normals_idx = np.where(domain.face_normals[2] != 0)[0]
            # top_bottom_faces = np.intersect1d(
            #     non_horizontal_normals_idx, all_bf
            # )  # all_bf = all_bf_idx
            # top = top_bottom_faces[
            #     face_centers[:, top_bottom_faces][2] < (self.zmin + self.zmax) / 2
            # ]
            # bottom = top_bottom_faces[
            #     face_centers[:, top_bottom_faces][2] > (self.zmin + self.zmax) / 2
            # ]

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

    def initial_condition(self) -> None:
        """ """
        # Zero for displacement and initial bc values for Biot
        super().initial_condition()

        # Contact as initial guess. Ensure traction is consistent with zero jump, which
        # follows from the default zeros set for all variables, specifically interface
        # displacement, by super method.
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

    def contact_mechanics_numerical_constant(
        self, subdomains: list[pp.Grid]
    ) -> pp.ad.Scalar:
        """ """
        # The constant works as a scaling factor in the comparison between tractions and
        # displacement jumps across fractures. In analogy with Hooke's law, the scaling
        # constant is therefore proportional to the shear modulus and the inverse of a
        # characteristic length of the fracture, where the latter has the interpretation
        # of a gradient length.

        shear_modulus = self.solid.shear_modulus()
        characteristic_distance = (
            self.solid.residual_aperture() + self.solid.fracture_gap()
        )

        # Physical interpretation (IS):
        # As a crude way of making the fracture softer than the matrix, we scale by
        # one order of magnitude.
        # Alternative interpretation (EK):
        # The scaling factor should not be too large, otherwise the contact problem
        # may be discretized wrongly. I therefore introduce a safety factor here; its
        # value is somewhat arbitrary.
        softening_factor = 1e-1

        val = softening_factor * shear_modulus / characteristic_distance

        return pp.ad.Scalar(val, name="Contact_mechanics_numerical_constant")

    def _is_nonlinear_problem(self) -> bool:
        """ """
        return self.mdg.dim_min() < self.nd

    def after_simulation(self) -> None:
        """ """

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
        # print("\n T_vect_frac = ", T_vect_frac)

        # pp.plot_grid(sd, vector_value=T_vect, figsize=(15, 12), alpha=0)
        # pp.plot_grid(
        #     self.sd_fract, T_vect_frac, alpha=0
        # )  # NO, for pp self.sd_fract is 2D, T_vect_frac is 3D, so they don't match, see below
        # T_vect_frac_filled = np.zeros((self.nd, sd.num_faces))
        # T_vect_frac_filled[:, self.fracture_faces_id] = T_vect_frac
        # pp.plot_grid(sd, vector_value=10000 * T_vect_frac_filled, alpha=0) # there is an eror in paraview... don't trust it

        exporter = pp.Exporter(self.sd_fract, "sd_fract")
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

        assert np.all(
            np.isclose(
                np.sqrt(T_tangential_norm**2 + T_normal_norm**2),
                np.linalg.norm(T_vect_frac.T, axis=1),
                rtol=0,
                atol=1e-8,
            )
        )

        exporter = pp.Exporter(model.mdg, file_name="eni_case", folder_name="./")
        exporter.write_vtu("u")


class FinalModel(
    MomentumBalanceEquations,
    VariablesMomentumBalance,
    ConstitutiveLawsMomentumBalance,
    BoundaryConditionsMomentumBalance,
    GeometryCloseToEni,
    SolutionStrategyMomentumBalance,
    pp.DataSavingMixin,
):
    pass


model = FinalModel()
pp.run_time_dependent_model(model, {})

# pp.plot_grid(
#     model.mdg,
#     vector_value=model.displacement_variable,
#     rgb=[1, 1, 1],
#     figsize=(10, 8),
#     linewidth=0.3,
#     title="Displacement",
# )

print("\nDone!")
