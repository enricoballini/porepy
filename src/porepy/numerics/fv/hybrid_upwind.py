import numpy as np
import scipy as sp
import porepy as pp
from . import hybrid_upwind_utils as hu_utils
from . import hybrid_weighted_average as hwa
import copy
import pdb


from porepy.numerics.discretization import Discretization


def myprint(var):
    print("\n" + var + " = ", eval(var))


"""
- procedurally paradigm is adopted
- I haven't specified "internal" to all the interal variables
"""


def flux_V(
    sd,
    mixture,
    ell,
    total_flux_internal,
    left_restriction,
    right_restriction,
    ad,
    dynamic_viscosity,
    mobility
):
    """ """

    def mobility_V_faces(
        saturation,
        total_flux_internal,
        left_restriction,
        right_restriction,
        dynamic_viscosity,
        mobility
    ):
        """ """

        mobility_upwinded = hu_utils.var_upwinded_faces(
            mobility(saturation, dynamic_viscosity),
            total_flux_internal,
            left_restriction,
            right_restriction,
        )
        return mobility_upwinded

    def mobility_tot_V_faces(
        saturation_list,
        total_flux_internal,
        left_restriction,
        right_restriction,
        ad,
        mobility
    ):
        """ """

        if ad:
            # mobility_tot = pp.ad.AdArray(
            #     np.zeros(left_restriction.shape[0]),
            #     0 * sp.sparse.eye(left_restriction.shape[0], 2 * sd.num_cells),
            # )  # TODO: find a smart way to initialize ad vars

            mobility_tot = (
                []
            )  # to initialize it you need the total number of dof, which is not a info sd related, therefore I avoid the initializazion and I append the elelnents in a list
        else:
            mobility_tot = np.zeros(
                left_restriction.shape[0], dtype=np.complex128
            )  # TODO: improve it ### this is not working anymore

        for m in np.arange(mixture.num_phases):
            mobility_tot.append(
                mobility_V_faces(
                    saturation_list[m],
                    total_flux_internal,
                    left_restriction,
                    right_restriction,
                    dynamic_viscosity,
                    mobility
                )
            )
            # I had to avoid "+=" bcs it requirs jac initialization. waiting for a better solution...

        mobility_tot = sum(mobility_tot)
        return mobility_tot

    # V (viscous/convective) flux computation:
    saturation_list = [None] * mixture.num_phases
    for phase_id in np.arange(mixture.num_phases):
        saturation_list[phase_id] = mixture.get_phase(phase_id).saturation

    mob_V = mobility_V_faces(
        saturation_list[ell],
        total_flux_internal,
        left_restriction,
        right_restriction,
        dynamic_viscosity,
        mobility
    )
    mob_tot_V = mobility_tot_V_faces(
        saturation_list,
        total_flux_internal,
        left_restriction,
        right_restriction,
        ad,
        mobility
    )
    V_internal = mob_V / mob_tot_V * total_flux_internal

    return V_internal


def rho_flux_V(
    sd,
    mixture,
    ell,
    pressure,
    total_flux_internal,
    left_restriction,
    right_restriction,
    ad,
    dynamic_viscosity,
    mobility
):
    """ """

    # 0D shortcut:
    if sd.dim == 0:
        # rho_V = pp.ad.AdArray(np.array([0]), 0*pressure.jac[0])
        rho_V = pp.ad.AdArray(np.empty((0)), sp.sparse.csr_matrix((0, pressure.jac.shape[1])))
        return rho_V

    V = pp.numerics.fv.hybrid_upwind.flux_V(
        sd,
        mixture,
        ell,
        total_flux_internal,
        left_restriction,
        right_restriction,
        ad,
        dynamic_viscosity,
        mobility
    )
    density = mixture.get_phase(ell).mass_density(pressure)
    rho_upwinded = hu_utils.var_upwinded_faces(
        density, V, left_restriction, right_restriction
    )
    rho_V_internal = rho_upwinded * V

    expansion = hu_utils.expansion_matrix(sd)
    rho_V = expansion @ rho_V_internal
    return rho_V


def flux_G(
    sd,
    mixture,
    ell,
    pressure,
    gravity_value,
    left_restriction,
    right_restriction,
    transmissibility_internal_tpfa,
    ad,
    dynamic_viscosity,
    dim_max,
    mobility
):
    """
    TODO: consider the idea to move omega outside the flux_G, if you don't see why => it is already in the right place.
    """

    def omega(num_phases, ell, mobilities, g, left_restriction, right_restriction, ad):
        """
        TODO: i run into some issues with pp.ad.functions.heaviside

        TODO: omega_ell doenst have to be a adArray
        """
        if ad:
            # omega_ell = pp.ad.AdArray(
            #     np.zeros(left_restriction.shape[0]),
            #     0 * sp.sparse.eye(left_restriction.shape[0], 2 * sd.num_cells),
            # )  # TODO: find a better way to initialize arrays
            omega_ell = []

            for m in np.arange(num_phases):
                omega_ell.append(
                    (
                        (left_restriction @ mobilities[m])
                        * pp.ad.functions.heaviside(-g[m] + g[ell])
                        + (right_restriction @ mobilities[m])
                        * pp.ad.functions.heaviside(g[m] - g[ell])
                    )
                    * (g[m] - g[ell])
                )

            omega_ell = sum(omega_ell)
        else:
            omega_ell = np.zeros(left_restriction.shape[0], dtype=np.complex128)

            for m in np.arange(num_phases):
                omega_ell += (
                    (left_restriction @ mobilities[m]) * (g[m] < g[ell])
                    + (right_restriction @ mobilities[m]) * (g[m] > g[ell])
                ) * (g[m] - g[ell])

        return omega_ell

    def mobility_G_faces(
        saturation,
        omega_ell,
        left_restriction,
        right_restriction,
        dynamic_viscosity,
        mobility,
    ):
        """ """
        mobility_upwinded = hu_utils.var_upwinded_faces(
            mobility(saturation, dynamic_viscosity),
            omega_ell,
            left_restriction,
            right_restriction,
        )
        return mobility_upwinded

    def mobility_tot_G_faces(
        num_phases,
        saturation_list,
        omega_ell,
        left_restriction,
        right_restriction,
        mobility,
    ):
        """ """
        if ad:
            # mobility_tot_G = pp.ad.AdArray(
            #     np.zeros(left_restriction.shape[0]),
            #     0 * sp.sparse.eye(left_restriction.shape[0], 2 * sd.num_cells),
            # )
            mobility_tot_G = []
        else:
            mobility_tot_G = np.zeros(left_restriction.shape[0], dtype=np.complex128)

        for m in np.arange(num_phases):  # m = phase_id
            mobility_tot_G.append(
                mobility_G_faces(
                    saturation_list[m],
                    omega_ell,
                    left_restriction,
                    right_restriction,
                    dynamic_viscosity,
                    mobility,
                )
            )

        mobility_tot_G = sum(mobility_tot_G)

        return mobility_tot_G

    # flux G computation:
    z = -sd.cell_centers[
        dim_max - 1
    ]  # zed is reversed to conform to paper 2022 notation

    saturation_list = [None] * mixture.num_phases
    g_list = [None] * mixture.num_phases
    mobility_list = [None] * mixture.num_phases
    omega_list = [None] * mixture.num_phases

    for phase_id in np.arange(mixture.num_phases):
        saturation = mixture.get_phase(phase_id).saturation  # ell and m ref paper
        saturation_list[phase_id] = saturation  # TODO: find a better solution
        rho = mixture.get_phase(phase_id).mass_density(pressure)
        rho = hu_utils.density_internal_faces(
            saturation, rho, left_restriction, right_restriction
        )  # TODO: rho used twice
        g_list[phase_id] = hu_utils.g_internal_faces(
            z, rho, gravity_value, left_restriction, right_restriction
        )  # TODO: g_ell and g_m are computed twice, one in G and one in omega
        mobility_list[phase_id] = mobility(saturation, dynamic_viscosity)

    for phase_id in np.arange(mixture.num_phases):
        omega_list[phase_id] = omega(
            mixture.num_phases,
            phase_id,
            mobility_list,
            g_list,
            left_restriction,
            right_restriction,
            ad,
        )

    mob_tot_G = mobility_tot_G_faces(
        mixture.num_phases,
        saturation_list,
        omega_list[ell],
        left_restriction,
        right_restriction,
        mobility,
    )

    if ad:
        # G_internal = pp.ad.AdArray(
        #     np.zeros(left_restriction.shape[0]),
        #     0 * sp.sparse.eye(left_restriction.shape[0], 2 * sd.num_cells),
        # )  # TODO: find a smart way to initialize ad vars
        G_internal = []
    else:
        G_internal = np.zeros(
            left_restriction.shape[0], dtype=np.complex128
        )  # TODO: improve it

    for m in np.arange(mixture.num_phases):
        mob_G_ell = mobility_G_faces(
            saturation_list[ell],
            omega_list[ell],
            left_restriction,
            right_restriction,
            dynamic_viscosity,
            mobility,
        )  # yes, you can move it outside the loop
        mob_G_m = mobility_G_faces(
            saturation_list[m],
            omega_list[m],
            left_restriction,
            right_restriction,
            dynamic_viscosity,
            mobility,
        )
        G_internal.append(mob_G_ell * mob_G_m / mob_tot_G * (g_list[m] - g_list[ell]))

    G_internal = sum(G_internal)
    G_internal *= transmissibility_internal_tpfa
    return G_internal


def rho_flux_G(
    sd,
    mixture,
    ell,
    pressure,
    gravity_value,
    left_restriction,
    right_restriction,
    transmissibility_internal_tpfa,
    ad,
    dynamic_viscosity,
    dim_max,
    mobility
):
    """ """

    # 0D shortcut:
    if sd.dim == 0:
        # rho_G = pp.ad.AdArray(np.array([0]), 0*pressure.jac[0])
        rho_G = pp.ad.AdArray(np.empty((0)), sp.sparse.csr_matrix((0, pressure.jac.shape[1])))
        return rho_G

    G = pp.numerics.fv.hybrid_upwind.flux_G(
        sd,
        mixture,
        ell,
        pressure,
        gravity_value,
        left_restriction,
        right_restriction,
        transmissibility_internal_tpfa,
        ad,
        dynamic_viscosity,
        dim_max,
        mobility
    )
    density = mixture.get_phase(ell).mass_density(pressure)
    rho_upwinded = hu_utils.var_upwinded_faces(
        density, G, left_restriction, right_restriction
    )
    rho_G_internal = rho_upwinded * G

    expansion = hu_utils.expansion_matrix(sd)
    rho_G = expansion @ rho_G_internal
    return rho_G


# def discretize(self, sd: pp.Grid, data: dict) -> None:
#     """ """

#     mixture = self.mixture
#     ell = self.ell
#     pressure = self.pressure
#     gravity_value = self.gravity_value
#     hu_utils.compute_transmissibility_tmp(sd, data)
#     transmissibility_internal_tpfa = hu_utils.utils.get_transmissibility_tpfa(sd, data)
#     left_restriction, right_restriction = hu_utils.restriction_matrices_left_right(sd)
#     ad = True
#     dynamic_viscosity = 1  # TODO: ...

#     hybridwa = hwa.HybridWeightedAverage()

#     total_flux_internal = hybridwa.total_flux()

#     rho_V_G = self.rho_flux_V(
#         sd,
#         mixture,
#         ell,
#         pressure,
#         total_flux_internal,
#         left_restriction,
#         right_restriction,
#         ad,
#         dynamic_viscosity,
#     ) + self.rho_flux_G(
#         sd,
#         mixture,
#         ell,
#         pressure,
#         gravity_value,
#         left_restriction,
#         right_restriction,
#         transmissibility_internal_tpfa,
#         ad,
#         dynamic_viscosity,
#     )


'''
def boundary_conditions_tmp(sd: pp.Grid, bc_val):
    """
    TODO: bc for pressure eq and for mass flux. For now, flux = 0, so I use this method for both

    TODO: bc_val is outwards flux wrt domain, not integrated. => if bc_val negative => entering mass
        change it(?) in according to pp convention

        TODO: not 100% sure about the sign, check it

        returns the rhs, in the right size and div included
    """
    # kind of div -abs(sg.cell_faces()).T@sd.face_areas()
    return -abs(sd.cell_faces).T @ (sd.face_areas * bc_val)


    @staticmethod
    def compute_jacobian_V_G_ad(
        sd, data, mixture, ell, pressure, gravity_value, ad, dynamic_viscosity
    ):
        """
        this is a tmp function and conceptually wrong. I momentary need  it to test the jac
        """
        L, R = HybridUpwind.restriction_matrices_left_right(sd)
        HybridUpwind.compute_transmissibility_tpfa(sd, data)
        _, transmissibility_internal_tpfa = HybridUpwind.get_transmissibility_tpfa(
            sd, data
        )

        qt_internal = HybridUpwind.total_flux(
            sd,
            mixture,
            pressure,
            gravity_value,
            L,
            R,
            transmissibility_internal_tpfa,
            ad,
            dynamic_viscosity,
        )
        qt_internal = qt_internal[0] + qt_internal[1]

        rho_V = HybridUpwind.rho_flux_V(
            sd, mixture, ell, pressure, qt_internal, L, R, ad, dynamic_viscosity
        )

        rho_G = HybridUpwind.rho_flux_G(
            sd,
            mixture,
            ell,
            pressure,
            gravity_value,
            L,
            R,
            transmissibility_internal_tpfa,
            ad,
            dynamic_viscosity,
        )

        pp_div = pp.fvutils.scalar_divergence(sd)

        rho_V_cell_no_bc = pp_div @ rho_V
        rho_G_cell_no_bc = pp_div @ rho_G

        flux_cell_no_bc = rho_V_cell_no_bc + rho_G_cell_no_bc

        return flux_cell_no_bc.val, flux_cell_no_bc.jac.A

    @staticmethod
    def compute_jacobian_V_G_complex(
        sd, data, mixture, pressure, ell, gravity_value, ad, dynamic_viscosity
    ):
        """attention: different logic wrt finite diff to add eps"""

        L, R = HybridUpwind.restriction_matrices_left_right(sd)
        HybridUpwind.compute_transmissibility_tpfa(sd, data)
        _, transmissibility_internal_tpfa = HybridUpwind.get_transmissibility_tpfa(
            sd, data
        )

        jacobian = np.zeros((sd.num_cells, 2 * sd.num_cells), dtype=np.complex128)
        eps = 1e-20j

        pp_div = pp.fvutils.scalar_divergence(sd)

        pressure_eps = copy.deepcopy(pressure)  # useless
        for i in np.arange(sd.num_cells):  # TODO: ...
            pressure_eps[i] += eps  ### ...

            qt = HybridUpwind.total_flux(
                sd,
                mixture,
                pressure_eps,
                gravity_value,
                L,
                R,
                transmissibility_internal_tpfa,
                ad,
                dynamic_viscosity,
            )
            qt = qt[0] + qt[1]

            V = HybridUpwind.rho_flux_V(
                sd, mixture, ell, pressure_eps, qt, L, R, ad, dynamic_viscosity
            )

            G = HybridUpwind.rho_flux_G(
                sd,
                mixture,
                ell,
                pressure_eps,
                gravity_value,
                L,
                R,
                transmissibility_internal_tpfa,
                ad,
                dynamic_viscosity,
            )

            F = V + G

            flux_cell_no_bc = pp_div @ F  # move it
            jacobian[:, i] = np.imag(flux_cell_no_bc) / np.imag(eps)

            pressure_eps[i] -= eps  ### ...

        saturation_ell = mixture.get_phase(ell).saturation
        saturation_ell_eps = copy.deepcopy(saturation_ell)  # useless

        if ell == 0:  # remove it... it is tmp
            m = 1
        else:
            m = 0
        saturation_m = mixture.get_phase(m).saturation
        saturation_m_eps = copy.deepcopy(saturation_m)

        for i in np.arange(sd.num_cells):  # TODO: ...
            saturation_ell_eps[i] += eps  ### ...
            mixture.get_phase(ell)._s = saturation_ell_eps

            saturation_m_eps[i] -= eps  # the constraint is here...
            mixture.get_phase(m)._s = saturation_m_eps

            qt = HybridUpwind.total_flux(
                sd,
                mixture,
                pressure,
                gravity_value,
                L,
                R,
                transmissibility_internal_tpfa,
                ad,
                dynamic_viscosity,
            )
            qt = qt[0] + qt[1]

            V = HybridUpwind.rho_flux_V(
                sd, mixture, ell, pressure, qt, L, R, ad, dynamic_viscosity
            )
            G = HybridUpwind.rho_flux_G(
                sd,
                mixture,
                ell,
                pressure,
                gravity_value,
                L,
                R,
                transmissibility_internal_tpfa,
                ad,
                dynamic_viscosity,
            )
            F = V + G

            flux_cell_no_bc = pp_div @ F
            jacobian[:, sd.num_cells + i] = np.imag(flux_cell_no_bc) / np.imag(
                eps
            )  # pay attention: jacobian[:,sd.num_cells+i]

            saturation_ell_eps[i] -= eps  ### ...
            saturation_m_eps[i] += eps  ### ...

        return jacobian

    @staticmethod
    def compute_jacobian_V_G_finite_diff(
        sd, data, mixture, pressure, ell, gravity_value, ad, dynamic_viscosity
    ):
        """attention: different logic wrt complex step to add eps"""

        L, R = HybridUpwind.restriction_matrices_left_right(sd)
        HybridUpwind.compute_transmissibility_tpfa(sd, data)
        _, transmissibility_internal_tpfa = HybridUpwind.get_transmissibility_tpfa(
            sd, data
        )

        jacobian = np.zeros((sd.num_cells, 2 * sd.num_cells), dtype=np.complex128)
        eps = 1e-5

        pp_div = pp.fvutils.scalar_divergence(sd)

        qt = HybridUpwind.total_flux(
            sd,
            mixture,
            pressure,
            gravity_value,
            L,
            R,
            transmissibility_internal_tpfa,
            ad,
            dynamic_viscosity,
        )
        qt = qt[0] + qt[1]

        V = HybridUpwind.rho_flux_V(
            sd, mixture, ell, pressure, qt, L, R, ad, dynamic_viscosity
        )
        G = HybridUpwind.rho_flux_G(
            sd,
            mixture,
            ell,
            pressure,
            gravity_value,
            L,
            R,
            transmissibility_internal_tpfa,
            ad,
            dynamic_viscosity,
        )
        F = V + G
        flux_cell_no_bc = pp_div @ F

        for i in np.arange(sd.num_cells):  # TODO: ...
            pressure_eps = copy.deepcopy(pressure)  ### ...
            pressure_eps[i] += eps  ### ...

            qt_eps = HybridUpwind.total_flux(
                sd,
                mixture,
                pressure_eps,
                gravity_value,
                L,
                R,
                transmissibility_internal_tpfa,
                ad,
                dynamic_viscosity,
            )
            qt_eps = qt_eps[0] + qt_eps[1]

            V_eps = HybridUpwind.rho_flux_V(
                sd, mixture, ell, pressure_eps, qt_eps, L, R, ad, dynamic_viscosity
            )
            G_eps = HybridUpwind.rho_flux_G(
                sd,
                mixture,
                ell,
                pressure_eps,
                gravity_value,
                L,
                R,
                transmissibility_internal_tpfa,
                ad,
                dynamic_viscosity,
            )
            F_eps = V_eps + G_eps

            flux_cell_no_bc_eps = pp_div @ F_eps
            jacobian[:, i] = (flux_cell_no_bc_eps - flux_cell_no_bc) / eps

        saturation_ell = mixture.get_phase(ell).saturation
        saturation_ell_eps = copy.deepcopy(saturation_ell)  ### TODO: ...

        if ell == 0:  # remove it... it is tmp
            m = 1
        else:
            m = 0
        saturation_m = mixture.get_phase(m).saturation
        saturation_m_eps = copy.deepcopy(saturation_m)

        for i in np.arange(sd.num_cells):  # TODO: ...
            saturation_ell_eps[i] += eps  ### ...
            mixture.get_phase(ell)._s = saturation_ell_eps

            saturation_m_eps[i] -= eps  # the constraint is here...
            mixture.get_phase(m)._s = saturation_m_eps

            qt_eps = HybridUpwind.total_flux(
                sd,
                mixture,
                pressure,
                gravity_value,
                L,
                R,
                transmissibility_internal_tpfa,
                ad,
                dynamic_viscosity,
            )
            qt_eps = qt_eps[0] + qt_eps[1]

            V_eps = HybridUpwind.rho_flux_V(
                sd, mixture, ell, pressure, qt_eps, L, R, ad, dynamic_viscosity
            )
            G_eps = HybridUpwind.rho_flux_G(
                sd,
                mixture,
                ell,
                pressure,
                gravity_value,
                L,
                R,
                transmissibility_internal_tpfa,
                ad,
                dynamic_viscosity,
            )
            F_eps = V_eps + G_eps

            flux_cell_no_bc_eps = pp_div @ F_eps
            jacobian[:, sd.num_cells + i] = (
                flux_cell_no_bc_eps - flux_cell_no_bc
            ) / eps  # pay attention: jacobian[:,sd.num_cells+i]

            saturation_ell_eps[i] -= eps  ### ...
            saturation_m_eps[i] += eps  ### ...

        return jacobian

    @staticmethod
    def assemble_matrix_rhs_ad(
        sd, data, mixture, pressure, ell, gravity_value, bc_val, ad, dynamic_viscosity
    ):
        """this function is conceptually wrong"""
        flux_qt, jacobian_qt = HybridUpwind.compute_jacobian_qt_ad(
            sd, data, mixture, pressure, gravity_value, ad, dynamic_viscosity
        )
        flux_V_G, jacobian_V_G = HybridUpwind.compute_jacobian_V_G_ad(
            sd, data, mixture, ell, pressure, gravity_value, ad, dynamic_viscosity
        )

        F = np.zeros(2 * sd.num_cells)
        F[0 : sd.num_cells] = flux_qt
        F[sd.num_cells : 2 * sd.num_cells] = flux_V_G

        JF = np.zeros((2 * sd.num_cells, 2 * sd.num_cells))
        JF[0 : sd.num_cells, 0 : 2 * sd.num_cells] = jacobian_qt
        JF[sd.num_cells : 2 * sd.num_cells, 0 : 2 * sd.num_cells] = jacobian_V_G

        b = HybridUpwind.boundary_conditions_tmp(sd, bc_val)

        return F, JF, b

    @staticmethod
    def assemble_matrix_rhs_complex(
        sd, data, mixture, pressure, ell, gravity_value, bc_val, ad, dynamic_viscosity
    ):
        """ """
        jacobian_qt = HybridUpwind.compute_jacobian_qt_complex(
            sd, data, mixture, pressure, ell, gravity_value, ad, dynamic_viscosity
        )
        jacobian_V_G = HybridUpwind.compute_jacobian_V_G_complex(
            sd, data, mixture, pressure, ell, gravity_value, ad, dynamic_viscosity
        )

        assert np.sum(np.imag(jacobian_qt)) == 0
        assert np.sum(np.imag(jacobian_V_G)) == 0

        jacobian_qt = np.real(jacobian_qt)
        jacobian_V_G = np.real(jacobian_V_G)

        A = np.zeros((2 * sd.num_cells, 2 * sd.num_cells))
        A[0 : sd.num_cells, 0 : 2 * sd.num_cells] = jacobian_qt
        A[sd.num_cells : 2 * sd.num_cells, 0 : 2 * sd.num_cells] = jacobian_V_G

        b = HybridUpwind.boundary_conditions_tmp(sd, bc_val)
        return A, b

    @staticmethod
    def assemble_matrix_rhs_tmp_finite_diff(
        sd, data, mixture, pressure, ell, gravity_value, bc_val, ad, dynamic_viscosity
    ):
        """ """
        jacobian_qt = HybridUpwind.compute_jacobian_qt_finite_diff(
            sd, data, mixture, pressure, ell, gravity_value, ad, dynamic_viscosity
        )
        jacobian_V_G = HybridUpwind.compute_jacobian_V_G_finite_diff(
            sd, data, mixture, pressure, ell, gravity_value, ad, dynamic_viscosity
        )

        assert np.sum(np.imag(jacobian_qt)) == 0
        assert np.sum(np.imag(jacobian_V_G)) == 0

        jacobian_qt = np.real(jacobian_qt)
        jacobian_V_G = np.real(jacobian_V_G)

        A = np.zeros((2 * sd.num_cells, 2 * sd.num_cells))
        A[0 : sd.num_cells, 0 : 2 * sd.num_cells] = jacobian_qt
        A[sd.num_cells : 2 * sd.num_cells, 0 : 2 * sd.num_cells] = jacobian_V_G

        b = HybridUpwind.boundary_conditions_tmp(sd, bc_val)
        return A, b

'''
