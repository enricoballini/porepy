"""
For any discretization class compatible with PorePy, wrap_discretization associates
a discretization with all attributes of the class' attributes that end with
'_matrix_key'.


Example:
    # Generate grid
    >>> g = pp.CartGrid([2, 2])
    # Associate an Ad representation of an Mpfa method, aimed this grid
    >>> discr = MpfaAd(keyword='flow', grids=[g])
    # The flux discretization of Mpfa can now be accesed by
    >>> discr.flux
    # While the discretization of boundary conditions is available by
    >>> discr.bound_flux.

    The representation of different discretization objects can be combined with other
    Ad objects into an operator tree, using lazy evaluation.

    It is assumed that the actual action of discretization (creation of the
    discretization matrices) is performed before the operator tree is parsed.
"""
from __future__ import annotations
import abc
from typing import List, Tuple, Union, Dict
import numpy as np
import scipy.sparse as sps

import porepy as pp

from ._ad_utils import MergedOperator, wrap_discretization

__all__ = [
    "Discretization",
    "BiotAd",
    "MpsaAd",
    "GradPAd",
    "DivUAd",
    "BiotStabilizationAd",
    "ColoumbContactAd",
    "ContactTractionAd",
    "MpfaAd",
    "TpfaAd",
    "MassMatrixAd",
    "UpwindAd",
    "RobinCouplingAd",
    "WellCouplingAd",
    "UpwindCouplingAd",
    "differentiable_mpfa",
]

Edge = Tuple[pp.Grid, pp.Grid]


class Discretization(abc.ABC):
    """General/utility methods for AD discretization classes.

    The init of the children classes below typically calls wrap_discretization
    and has arguments including grids or edges and keywords for parameter and
    possibly matrix storage.

    """

    def __init__(self):
        """"""

        self._discretization: Union[
            "pp.numerics.discretization.Discretization",
            "pp.numerics.interface_laws.abstract_interface_law.AbstractInterfaceLaw",
        ]
        self.mat_dict_key: str
        self.keyword = str

        # Get the name of this discretization.
        self._name: str
        self.grids: List[pp.Grid]
        self.edges: List[Edge]

    def __repr__(self) -> str:
        s = f"Ad discretization of type {self._name}. Defined on {len(self.grids)} grids"
        return s

    def __str__(self) -> str:
        return f"{self._name}({self.keyword})"


### Mechanics related discretizations


class BiotAd(Discretization):
    """Ad wrapper around the Biot discretization class.

    For description of the method, we refer to the standard Biot class.

    """

    def __init__(self, keyword: str, grids: List[pp.Grid]) -> None:
        self.grids = grids
        self._discretization = pp.Biot(keyword)
        self._name = "BiotMpsa"

        self.keyword = keyword

        # Declear attributes, these will be initialized by the below call to the
        # discretization wrapper.

        self.stress: MergedOperator
        self.bound_stress: MergedOperator
        self.bound_displacement_cell: MergedOperator
        self.bound_displacement_face: MergedOperator

        self.div_u: MergedOperator
        self.bound_div_u: MergedOperator
        self.grad_p: MergedOperator
        self.stabilization: MergedOperator
        self.bound_pressure: MergedOperator

        wrap_discretization(
            obj=self, discr=self._discretization, grids=grids, mat_dict_key=self.keyword
        )


class MpsaAd(Discretization):
    def __init__(self, keyword: str, grids: List[pp.Grid]) -> None:
        self.grids = grids
        self._discretization = pp.Mpsa(keyword)
        self._name = "Mpsa"

        self.keyword = keyword

        # Declear attributes, these will be initialized by the below call to the
        # discretization wrapper.

        self.stress: MergedOperator
        self.bound_stress: MergedOperator
        self.bound_displacement_cell: MergedOperator
        self.bound_displacement_face: MergedOperator

        wrap_discretization(self, self._discretization, grids=grids)


class GradPAd(Discretization):
    def __init__(self, keyword: str, grids: List[pp.Grid]) -> None:
        self.grids = grids
        self._discretization = pp.GradP(keyword)
        self._name = "GradP from Biot"
        self.keyword = keyword

        self.grad_p: MergedOperator

        wrap_discretization(self, self._discretization, grids=grids)


class DivUAd(Discretization):
    def __init__(
        self, keyword: str, grids: List[pp.Grid], mat_dict_keyword: str
    ) -> None:
        self.grids = grids
        self._discretization = pp.DivU(keyword, mat_dict_keyword)

        self._name = "DivU from Biot"
        self.keyword = mat_dict_keyword

        self.div_u: MergedOperator
        self.bound_div_u: MergedOperator

        wrap_discretization(
            self, self._discretization, grids=grids, mat_dict_key=mat_dict_keyword
        )


class BiotStabilizationAd(Discretization):
    def __init__(self, keyword: str, grids: List[pp.Grid]) -> None:
        self.grids = grids
        self._discretization = pp.BiotStabilization(keyword)
        self._name = "Biot stabilization term"
        self.keyword = keyword

        self.stabilization: MergedOperator

        wrap_discretization(self, self._discretization, grids=grids)


class ColoumbContactAd(Discretization):
    def __init__(self, keyword: str, edges: List[Edge]) -> None:
        self.edges = edges

        # Special treatment is needed to cover the case when the edge list happens to
        # be empty.
        if len(edges) > 0:
            dim = np.unique([e[0].dim for e in edges])

            low_dim_grids = [e[1] for e in edges]
            if not dim.size == 1:
                raise ValueError(
                    "Expected unique dimension of grids with contact problems"
                )
        else:
            # The assigned dimension value should never be used for anything, so we
            # set a negative value to indicate this (not sure how the parameter is used)
            # in the real contact discretization.
            dim = [-1]
            low_dim_grids = []

        self._discretization = pp.ColoumbContact(
            keyword, ambient_dimension=dim[0], discr_h=pp.Mpsa(keyword)
        )
        self._name = "Coloumb contact"
        self.keyword = keyword

        self.traction: MergedOperator
        self.displacement: MergedOperator
        self.rhs: MergedOperator
        wrap_discretization(
            self, self._discretization, edges=edges, mat_dict_grids=low_dim_grids
        )


class ContactTractionAd(Discretization):
    def __init__(self, keyword: str, edges: List[Edge]) -> None:
        self.edges = edges

        # Special treatment is needed to cover the case when the edge list happens to
        # be empty.
        if len(edges) > 0:
            dim = np.unique([e[0].dim for e in edges])

            low_dim_grids = [e[1] for e in edges]
            if not dim.size == 1:
                raise ValueError(
                    "Expected unique dimension of grids with contact problems"
                )
        else:
            # The assigned dimension value should never be used for anything, so we
            # set a negative value to indicate this (not sure how the parameter is used)
            # in the real contact discretization.
            dim = [-1]
            low_dim_grids = []

        self._discretization = pp.ContactTraction(
            keyword, ambient_dimension=dim, discr_h=pp.Mpsa(keyword)
        )
        self._name = "Simple ad contact"
        self.keyword = keyword

        self.normal: MergedOperator
        self.tangential: MergedOperator
        self.traction_scaling: MergedOperator

        wrap_discretization(
            self, self._discretization, edges=edges, mat_dict_grids=low_dim_grids
        )


## Flow related


class MpfaAd(Discretization):
    def __init__(self, keyword: str, grids: List[pp.Grid]) -> None:
        self.grids = grids
        self._discretization = pp.Mpfa(keyword)
        self._name = "Mpfa"
        self.keyword = keyword

        self.flux: MergedOperator
        self.bound_flux: MergedOperator
        self.bound_pressure_cell: MergedOperator
        self.bound_pressure_face: MergedOperator
        self.vector_source: MergedOperator
        self.bound_pressure_vector_source: MergedOperator

        wrap_discretization(self, self._discretization, grids=grids)


class TpfaAd(Discretization):
    def __init__(self, keyword: str, grids: List[pp.Grid]) -> None:
        self.grids = grids
        self._discretization = pp.Tpfa(keyword)
        self._name = "Tpfa"
        self.keyword = keyword

        self.flux: MergedOperator
        self.bound_flux: MergedOperator
        self.bound_pressure_cell: MergedOperator
        self.bound_pressure_face: MergedOperator
        self.vector_source: MergedOperator
        self.bound_pressure_vector_source: MergedOperator

        wrap_discretization(self, self._discretization, grids=grids)


class MassMatrixAd(Discretization):
    def __init__(self, keyword: str, grids: List[pp.Grid]) -> None:
        self.grids = grids
        self._discretization = pp.MassMatrix(keyword)
        self._name = "Mass matrix"
        self.keyword = keyword

        self.mass: MergedOperator
        wrap_discretization(self, self._discretization, grids=grids)


class UpwindAd(Discretization):
    def __init__(self, keyword: str, grids: List[pp.Grid]) -> None:
        self.grids = grids
        self._discretization = pp.Upwind(keyword)
        self._name = "Upwind"
        self.keyword = keyword

        self.upwind: MergedOperator
        self.bound_transport_dir: MergedOperator
        self.bound_transport_neu: MergedOperator
        wrap_discretization(self, self._discretization, grids=grids)


## Interface coupling discretizations


class WellCouplingAd(Discretization):
    def __init__(self, keyword: str, edges: List[Edge]) -> None:
        self.edges = edges
        self._discretization = pp.WellCoupling(keyword, primary_keyword=keyword)
        self._name = "Well interface coupling"
        self.keyword = keyword

        self.well_discr: MergedOperator
        self.well_vector_source: MergedOperator
        wrap_discretization(self, self._discretization, edges=edges)

    def __repr__(self) -> str:
        s = (
            f"Ad discretization of type {self._name}."
            f"Defined on {len(self.edges)} mortar grids."
        )
        return s


class RobinCouplingAd(Discretization):
    def __init__(self, keyword: str, edges: List[Edge]) -> None:
        self.edges = edges
        self._discretization = pp.RobinCoupling(keyword, primary_keyword=keyword)
        self._name = "Robin interface coupling"
        self.keyword = keyword

        self.mortar_discr: MergedOperator
        self.mortar_vector_source: MergedOperator
        wrap_discretization(self, self._discretization, edges=edges)

    def __repr__(self) -> str:
        s = (
            f"Ad discretization of type {self._name}."
            f"Defined on {len(self.edges)} mortar grids."
        )
        return s


class UpwindCouplingAd(Discretization):
    def __init__(self, keyword: str, edges: List[Edge]) -> None:
        self.edges = edges
        self._discretization = pp.UpwindCoupling(keyword)
        self._name = "Upwind coupling"
        self.keyword = keyword

        # UpwindCoupling also has discretization matrices for (inverse) trace.
        # These are not needed for Ad version since ad.Trace should be used instead
        self.mortar_discr: MergedOperator
        self.flux: MergedOperator
        self.upwind_primary: MergedOperator
        self.upwind_secondary: MergedOperator
        wrap_discretization(self, self._discretization, edges=edges)

    def __repr__(self) -> str:
        s = (
            f"Ad discretization of type {self._name}."
            f"Defined on {len(self.edges)} mortar grids."
        )
        return s


def differentiable_mpfa(
    perm_function: pp.ad.Function,
    perm_argument: pp.ad.Variable,
    potential: pp.ad.Variable,
    grid_list: List[pp.Grid],
    bc: Dict[pp.Grid, pp.BoundaryCondition],
    base_discr: Union[pp.ad.MpfaAd, pp.ad.TpfaAd],
    dof_manager: pp.DofManager,
    var_name: str,
    projections: pp.ad.SubdomainProjections,
) -> pp.ad.Ad_array:
    """
    This function applies the product and chain rule of the flux expression

        q = T(k(u)) * p

    Where the transmissibility matrix T is a function of the cell permeability k, which
    again is a function of a primary variable u, while p is the potential (pressure).

    Parameters:
        perm_function (pp.ad.Function): Function which gives the permeability as a
            function of primary variables.
        potential (pp.ad.Variable): Variable that enters the diffusion law (pressure, say).
        bc (pp.BoundaryCondition):
        base_discr: Tpfa or Mpfa discretization (Ad), gol which we want to approximate
            the transmissibility matrix.
        dof_manager (pp.DofManager): Needed to evaluate Ad operators.
        var_name (str): Name of the potential variable (as known to the DofManager).

    Returns:
        Ad_array: The flux, q, and its Jacobian matrix, where the latter accounts for
            dependencies in the transmissibilities on cell center permeabilities.

    """
    # Note on parameters: When this function is called, potential should be an Ad
    # operator (say, a MergedVariable representation of the pressure). During evaluation,
    # because of the way operator trees are evaluated, potential will be an Ad_array
    # (it is closer to being an atomic variable, thus it will be evaualted before
    # this function).

    # Mypy will not like this.
    # The product rule applied to q = T(k(u)) * p gives
    #   dT/dk * dk/du * p + T * dp/dp.
    # The first part is rather involved.

    # Get hold of the underlying flux discretization.
    base_flux = base_discr.flux.evaluate(dof_manager)
    # The Jacobian matrix should have the same size as the base.
    block_jac = sps.csr_matrix(base_flux.shape)

    # The differentiation of transmissibilities with respect to permeability is
    # implemented as a loop over all grids. It could be possible to gather all grid
    # information in arrays as a preprocessing step, but that seems not to be worth
    # the effort.
    for g in grid_list:

        # The first few lines are pasted from the standard Tpfa implementation
        fi, ci, sgn = sps.find(g.cell_faces)
        sz = fi.size

        # Normal vectors and permeability for each face (here and there side)
        n = g.face_normals[:, fi]
        # Switch signs where relevant
        n *= sgn

        # Distance from face center to cell center
        fc_cc = g.face_centers[::, fi] - g.cell_centers[::, ci]

        # This is really the expression n * K * dist(x_face, x_cell), but since we assume
        # the permeability is isotropic, we deal with this below.
        n_dist = n * fc_cc
        dist_face_cell = np.power(fc_cc, 2).sum(axis=0)

        # From here on, the code is specific for transmissibility differentiation.

        # The chain rule applied to T(k(u)) * p (where the k(u) dependency can be replaced
        # by other primary variables - Ad should take care of this) gives
        #
        #    dT/du * p + T dp =  * p + T dp
        #
        # Here, dT/du is in reality a third-order tensor, which we have represented as a matrix
        # (assuming isotropic permeability). For simplicity of implementation, we move the
        # gradient from dT/du to the p term. This implies as far as EK/IS have deduced that
        # each line of dT/du (# faces x # global dofs) should be multiplied with the gradient
        # at the corresponding face.
        # The chain rule gives
        #
        #   dT/du = dT/dk dk/du.
        #
        # k being an ad function, dk/du is available as .jac after evaluation. dT/dk is computed
        # below. Its deduction requires differentiating T w.r.t. half transmissibilities and
        # collection of contributions from each cell. For i in {left, right} cells, it reads
        #
        #   dT_face/dt_i = T_face ** 2 / t_i ** 2,
        #
        # where the tpfa face transmissibility is
        #
        #   T_face = 1 / (1/t_l + 1/t_r).

        # Evaluate the permeability as a function of the current potential
        # The evaluation means we go from an Ad operator formulation to the forward mode,
        # working with Ad_arrays. We map the computed permeability to the faces (distinguishing between
        # the left and right sides of the face).
        cell_2_one_sided_face = sps.coo_matrix(
            (np.ones(sz), (np.arange(sz), ci)),
            shape=(sz, g.num_cells),
        ).tocsr()

        # Evaluate the permeability as a function of the current potential and restrict
        # the permeability to the current grid (if perm_function is specific to the
        # grid, the restriction should be applied to the potential rather than the
        # permeability).
        # The evaluation means we go from an Ad operator formulation to the forward mode,
        # working with Ad_arrays.
        # Finally map the computed permeability to the faces (distinguishing between
        # the left and right sides of the face).
        k_one_sided = cell_2_one_sided_face * perm_function(perm_argument).evaluate(
            dof_manager
        )
        #        k_one_sided = cell_2_one_sided_face * (
        #            (projections.cell_restriction(g)) * perm_function(perm_argument)
        #        ).evaluate(dof_manager)

        # Multiply the permeability (and its derivatives with respect to potential,
        # since k_one_sided is an Ad_array) with normal vectors divided by distance
        t_one_sided = (
            sps.dia_matrix(
                (np.divide(n_dist.sum(axis=0), dist_face_cell), 0), shape=(sz, sz)
            )
            * k_one_sided
        )

        # Mapping which sums the right and left sides of the face.
        # Unlike in normal tpfa, the sign of the normal vector is disregarded.
        # This is made up for when multiplying by grad p.
        sum_cell_face_pair_to_face = sps.coo_matrix(
            (np.ones(sz), (fi, np.arange(sz))), shape=(g.num_faces, sz)
        ).tocsr()

        # Compute the two factors of dT_face/dt_i (see definition and explanation above).
        inverse_sum_squared = (
            sum_cell_face_pair_to_face * ((1 / t_one_sided.val))
        ) ** 2

        face_transmissibility_squared = sps.dia_matrix(
            (1 / inverse_sum_squared, 0), shape=(g.num_faces, g.num_faces)
        )
        hf_vals = np.power(t_one_sided.val, -2)
        half_face_transmissibility_inv_squared = sps.coo_matrix(
            (hf_vals, (fi, np.arange(sz))), shape=(g.num_faces, sz)
        ).tocsr()
        d_transmissibility_d_k = (
            face_transmissibility_squared * half_face_transmissibility_inv_squared
        )

        num_glob_dofs = dof_manager.num_dofs()

        # Use unit values for elements not associated with the potential in this grid
        # (unit values seem more natural than zeros here, since the latter risks
        # eliminating columns relevant for problems where the permeability is a function
        # of other variables than the potential).
        loc_vals = np.ones(num_glob_dofs)

        # Potential for this grid (Could also be retrieved using potential.evaluate?)
        potential_value = dof_manager.assemble_variable(
            grids=[g], variables=[var_name], from_iterate=True
        )
        # Create matrix and multiply into Jacobian
        grad_p = sps.diags(
            pp.fvutils.scalar_divergence(g).T * potential_value,
            shape=(g.num_faces, g.num_faces),
        )
        jac = grad_p * d_transmissibility_d_k * k_one_sided.jac

        # Eliminate values on Neumann boundaries.
        # FIXME: we should a corresponding operation for Dirichlet.
        is_neu = bc[g].is_neu
        pp.matrix_operations.zero_rows(jac, np.where(is_neu)[0])

        # Prolong this Jacobian to the full set of faces and add.
        block_jac += projections.face_prolongation(g).evaluate(dof_manager) * jac

    # Second part of product rule, applied to the potential. This is the standard part of
    # a Mpfa (Tpfa) discretization
    block_jac += base_flux * potential.jac

    # The value of the flux is the standard mpfa/tpfa expression.
    block_val = base_flux * potential.val

    return pp.ad.Ad_array(block_val, block_jac)
