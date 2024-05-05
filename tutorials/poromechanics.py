import os
import sys
import pdb
import inspect
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
            data = np.zeros((sd.num_cells, self.nd))

            if sd.dim == 2:
                # Selecting central cells
                cell_centers = sd.cell_centers
                indices = (
                    (cell_centers[0] > (0.3 / units.m))
                    & (cell_centers[0] < (0.7 / units.m))
                    & (cell_centers[1] > (0.3 / units.m))
                    & (cell_centers[1] < (0.7 / units.m))
                )

                acceleration = self.solid.convert_units(-9.8, "m * s^-2")
                force = self.solid.density() * acceleration
                data[indices, 1] = force * sd.cell_volumes[indices]

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
        """ """
        return np.zeros((self.nd, boundary_grid.num_cells)).ravel("F")

    def bc_values_stress(self, boundary_grid: pp.BoundaryGrid) -> np.ndarray:
        """ """
        return np.zeros((self.nd, boundary_grid.num_cells)).ravel("F")

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
        # cell_size_y = self.solid.convert_units(0.005, "m")
        cell_size_x = self.solid.convert_units(0.2, "m")
        cell_size_y = self.solid.convert_units(0.2, "m")
        cell_size_z = self.solid.convert_units(0.02, "m")
        meshing_params = {
            "cell_size_x": cell_size_x,
            "cell_size_y": cell_size_y,
            "cell_size_z": cell_size_z,
        }

        self.set_domain()
        self.set_fractures()
        # Create a fracture network.
        self.fracture_network = pp.create_fracture_network(self.fractures, self.domain)

        # self.mdg = pp.create_mdg(
        #     grid_type="cartesian",
        #     meshing_args=meshing_params,
        #     fracture_network=self.fracture_network,
        #     **self.meshing_kwargs(),
        # )

        eni_grid = self.load_eni_grid(
            path_to_mat="/home/inspiron/Desktop/PhD/7-eni/enricoeni/Tesi/mrst-2023b/mrst_grid.mat"
        )

        self.mdg = pp.MixedDimensionalGrid()
        self.mdg.add_subdomains(eni_grid)
        self.mdg.compute_geometry()  # let's do it another time...

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

        # pp.plot_grid(self.mdg, alpha=0, info="f")

    def set_domain(self) -> None:
        """ """
        box = {"xmin": 0, "xmax": 10, "ymin": 0, "ymax": 1, "zmin": 0, "zmax": 1}
        self._domain = pp.domain.Domain(bounding_box=box)

    def set_fractures(self) -> None:
        """ """
        # [[x_min, x_max],[y_min, y_max]]
        # frac_1 = pp.LineFracture(np.array([[3, 3], [0, 0.5]]))

        # [[pt1], [pt2], [], []].T
        frac_1 = pp.PlaneFracture(
            np.array([[3, 0, 0], [3, 1, 0], [3, 1, 0.5], [3, 0, 0.5]]).T
        )
        self._fractures = [frac_1]

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


class FinalModel(
    MomentumBalanceEquations,
    VariablesMomentumBalance,
    ConstitutiveLawsMomentumBalance,
    BoundaryConditionsMomentumBalance,
    SolutionStrategyMomentumBalance,
    GeometryCloseToEni,
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

exporter = pp.Exporter(model.mdg, file_name="TMP", folder_name="./")
exporter.write_vtu()

print("\Done!")
pdb.set_trace()

#
#
#
#
#
#
#
#
#
#
#
#
#
#
#
#
#
#
#
#
#
#
#
#
#
#
#
#
#
#
#
#


class PressureSourceBC:
    def bc_type_darcy_flux(self, sd: pp.Grid) -> pp.BoundaryCondition:
        """Assign Dirichlet boundary condition to the north boundary and Neumann
        everywhere else.

        """
        bounds = self.domain_boundary_sides(sd)
        bc = pp.BoundaryCondition(sd, bounds.north, "dir")
        return bc

    def fluid_source(self, subdomains: list[pp.Grid]) -> pp.ad.Operator:
        """Assign fracture source."""
        # Retrieve internal sources (jump in mortar fluxes) from the base class
        internal_sources: pp.ad.Operator = super().fluid_source(subdomains)

        # Retrieve external (integrated) sources from the exact solution.
        values = []
        src_value: float = self.fluid.convert_units(0.1, "kg * m^-3 * s^-1")
        for sd in subdomains:
            if sd.dim == self.mdg.dim_max():
                values.append(np.zeros(sd.num_cells))
            else:
                values.append(np.ones(sd.num_cells) * src_value)

        external_sources = pp.wrap_as_ad_array(np.concatenate(values))

        # Add up both contributions
        source = internal_sources + external_sources
        source.set_name("fluid sources")

        return source


from porepy.models.fluid_mass_balance import SinglePhaseFlow


class PoromechanicsSourceBC(
    PressureSourceBC,
    SquareDomainOrthogonalFractures,
    SinglePhaseFlow,
):
    """Adding geometry, boundary conditions and source to the default model."""

    def meshing_arguments(self) -> dict:
        cell_size = self.solid.convert_units(0.1, "m")
        return {"cell_size": cell_size}


model = PoromechanicsSourceBC()
pp.run_time_dependent_model(model, {})
pp.plot_grid(
    model.mdg,
    cell_value=model.pressure_variable,
    figsize=(10, 8),
    linewidth=0.25,
    title="Pressure field",
)


from porepy.models.poromechanics import Poromechanics


class PoromechanicsSourceBC(
    PressureSourceBC,
    BodyForceMixin,
    SquareDomainOrthogonalFractures,
    Poromechanics,
):
    """Adding geometry, boundary conditions and source to the default model."""

    def meshing_arguments(self) -> dict:
        cell_size = self.solid.convert_units(0.1, "m")
        return {"cell_size": cell_size}


model = PoromechanicsSourceBC()
pp.run_time_dependent_model(model, {})
pp.plot_grid(
    model.mdg,
    cell_value=model.pressure_variable,
    vector_value=model.displacement_variable,
    figsize=(10, 8),
    title="Pressure and displacement",
)


# You may notice that the base class `Poromechanics` is defined by combining several mixins:


import inspect


print(inspect.getsource(Poromechanics))


# Each of these mixins is a combination of the flow and the mechanics mixins. For instance:

from porepy.models.poromechanics import EquationsPoromechanics


print(inspect.getsource(EquationsPoromechanics))


# Thereby, the philosophy of multiple inheritance and mixins helps to modularize and reuse
#  a significant part of code.

# # What we have explored
# We set up and ran a poromechanics simulation using the body force and source mixin classes originally designed for uncoupled problems.
