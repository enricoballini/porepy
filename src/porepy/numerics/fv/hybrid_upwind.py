import numpy as np
import scipy as sp
import porepy as pp
from typing import List, Union, Callable
from . import hybrid_upwind_utils as hu_utils
from . import hybrid_weighted_average as hwa
import copy
import pdb


from porepy.numerics.discretization import Discretization


def myprint(var):
    print("\n" + var + " = ", eval(var))


"""
- procedurally paradigm is adopted
- hu works only on internal faces, boundary condistions (only homogeneous neumann so far) are applied somewhere else in the model
- I haven't specified the subscript "internal" to all the interal variables
- complex step was implemented along with ad. Results showed same accuracy even across discontinuities. Never compared the time efficiency.
"""


def flux_V(
    sd: pp.Grid,
    mixture: pp.Mixture,
    ell: int,
    total_flux_internal: pp.ad.AdArray,
    left_restriction: sp.sparse.spmatrix,
    right_restriction: sp.sparse.spmatrix,
    ad: bool,
    dynamic_viscosity: float,
    mobility: Callable,
) -> pp.ad.AdArray:
    """ """

    def mobility_V_faces(
        saturation,
        total_flux_internal,
        left_restriction,
        right_restriction,
        dynamic_viscosity,
        mobility,
    ) -> pp.ad.AdArray:
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
        mobility,
    ) -> pp.ad.AdArray:
        """ """

        if ad:
            mobility_tot = (
                []
            )  # to initialize it you need the total number of dof, which is not a info sd related, therefore I avoid the initializazion and I append the elements in a list
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
                    mobility,
                )
            )

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
        mobility,
    )
    mob_tot_V = mobility_tot_V_faces(
        saturation_list,
        total_flux_internal,
        left_restriction,
        right_restriction,
        ad,
        mobility,
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
    mobility,
) -> pp.ad.AdArray:
    """ """

    # 0D shortcut:
    if sd.dim == 0:
        rho_V = pp.ad.AdArray(
            np.empty((0)), sp.sparse.csr_matrix((0, pressure.jac.shape[1]))
        )
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
        mobility,
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
    sd: pp.Grid,
    mixture: pp.Mixture,
    ell: int,
    pressure: pp.ad.AdArray,
    gravity_value: float,
    left_restriction: sp.sparse.spmatrix,
    right_restriction: sp.sparse.spmatrix,
    transmissibility_internal_tpfa,
    ad: bool,
    dynamic_viscosity: np.ndarray,
    dim_max: int,
    mobility: Callable,
) -> pp.ad.AdArray:
    """ """

    def omega(
        num_phases, ell, mobilities, g, left_restriction, right_restriction, ad
    ) -> Union[pp.ad.AdArray, np.ndarray]:
        """ """
        if ad:
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
    ) -> pp.ad.AdArray:
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
    ) -> pp.ad.AdArray:
        """ """
        if ad:
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
        saturation = mixture.get_phase(phase_id).saturation  # ell and m ref paper Hamon
        saturation_list[phase_id] = saturation
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
        G_internal = []
    else:
        G_internal = np.zeros(left_restriction.shape[0], dtype=np.complex128)

    mob_G_ell = mobility_G_faces(
        saturation_list[ell],
        omega_list[ell],
        left_restriction,
        right_restriction,
        dynamic_viscosity,
        mobility,
    )
    for m in np.arange(mixture.num_phases):
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
    mobility,
) -> pp.ad.AdArray:
    """ """

    # 0D shortcut:
    if sd.dim == 0:
        rho_G = pp.ad.AdArray(
            np.empty((0)), sp.sparse.csr_matrix((0, pressure.jac.shape[1]))
        )
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
        mobility,
    )
    density = mixture.get_phase(ell).mass_density(pressure)
    rho_upwinded = hu_utils.var_upwinded_faces(
        density, G, left_restriction, right_restriction
    )
    rho_G_internal = rho_upwinded * G

    expansion = hu_utils.expansion_matrix(sd)
    rho_G = expansion @ rho_G_internal
    return rho_G


def omega(
    num_phases, ell, mobilities, g, left_restriction, right_restriction, ad
) -> pp.ad.AdArray:
    """
    copied from flux G. Need it in model to compute number of flips.
    """
    if ad:
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

    return omega_ell
