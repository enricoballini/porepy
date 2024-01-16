import numpy as np
import scipy as sp
import porepy as pp
from typing import Callable, Tuple, Union
from . import hybrid_upwind_utils as hu_utils
import copy
import pdb


from porepy.numerics.discretization import Discretization


def myprint(var):
    print("\n" + var + " = ", eval(var))


def ndof(g: pp.Grid) -> int:
    """ """
    # hardcoded for two-phase
    return 2 * g.num_cells


def total_flux_internal(
    sd: pp.Grid,
    mixture: pp.Mixture,
    pressure: pp.ad.AdArray,
    gravity_value: float,
    left_restriction: sp.sparse.spmatrix,
    right_restriction: sp.sparse.spmatrix,
    transmissibility_internal_tpfa: np.ndarray,
    ad: bool,
    dynamic_viscosity: float,
    dim_max: int,
    mobility: Callable,
    permeability: Callable,
) -> pp.ad.AdArray:
    """ """

    def gamma_value(permeability):
        """ """
        alpha = 1.0  # as in the paper 2022

        kr0 = permeability(saturation=1)

        def second_derivative(permeability, val):
            """ """
            h = 1e-4
            return (
                permeability(val + h) - 2 * permeability(val) + permeability(val - h)
            ) / (h**2)

        dd_kr_max = np.nanmax(
            second_derivative(permeability, np.linspace(0, 1, 10))
        )  # TODO: improve it...

        gamma_val = alpha / kr0 * dd_kr_max
        return gamma_val

    def g_ref_faces(
        mixture: pp.Mixture,
        pressure: pp.ad.AdArray,
        z: np.ndarray,
        gravity_value: float,
        left_restriction: sp.sparse.spmatrix,
        right_restriction: sp.sparse.spmatrix,
    ):
        """ """
        # HARDCODED for 2 phase
        density_faces_0 = hu_utils.density_internal_faces(
            mixture.get_phase(0).saturation,
            mixture.get_phase(0).mass_density(pressure),
            left_restriction,
            right_restriction,
        )
        density_faces_1 = hu_utils.density_internal_faces(
            mixture.get_phase(1).saturation,
            mixture.get_phase(1).mass_density(pressure),
            left_restriction,
            right_restriction,
        )

        density_max = pp.ad.maximum(density_faces_0, density_faces_1)

        g_ref = (
            density_max * gravity_value * (left_restriction @ z - right_restriction @ z)
        )
        return g_ref

    def delta_potential_faces(
        pressure,
        saturation,
        density,
        z,
        gravity_value,
        left_restriction,
        right_restriction,
    ):
        """ """
        density_internal_faces = hu_utils.density_internal_faces(
            saturation, density, left_restriction, right_restriction
        )
        g_internal_faces = hu_utils.g_internal_faces(
            z,
            density_internal_faces,
            gravity_value,
            left_restriction,
            right_restriction,
        )
        delta_pot = (
            left_restriction @ pressure - right_restriction @ pressure
        ) - g_internal_faces

        return delta_pot

    def beta_faces(
        pressure,
        saturation,
        density,
        z,
        gravity_value,
        gamma_val,
        g_ref_faces,
        left_restriction,
        right_restriction,
        ad,
    ):
        """ """
        c_faces_ref = 0  # no capillary pressure

        delta_pot_faces = delta_potential_faces(
            pressure,
            saturation,
            density,
            z,
            gravity_value,
            left_restriction,
            right_restriction,
        )
        tmp = gamma_val / (
            pp.ad.abs(g_ref_faces) + c_faces_ref + 1e-8
        )  # added epsilon to avoid division by zero

        if ad:
            tmp = -pp.ad.functions.maximum(-tmp, -1e6)
            beta_faces = 0.5 + 1 / np.pi * pp.ad.arctan(tmp * delta_pot_faces)
        else:
            tmp = np.minimum(np.real(tmp), 1e6) + np.imag(tmp) * 1j
            beta_faces = 0.5 + 1 / np.pi * np.arctan(tmp * delta_pot_faces)

        return beta_faces

    def lambda_WA_faces(beta_faces, mobility, left_restriction, right_restriction):
        """ """
        lambda_WA = beta_faces * (left_restriction @ mobility) + (1 - beta_faces) * (
            right_restriction @ mobility
        )
        return lambda_WA

    # total flux computation:

    # 0D shortcut:
    if sd.dim == 0:
        total_flux = [None] * mixture.num_phases
        for m in np.arange(mixture.num_phases):
            total_flux[m] = pp.ad.AdArray(
                np.empty((0)), sp.sparse.csr_matrix((0, pressure.jac.shape[1]))
            )

        return total_flux

    z = -sd.cell_centers[
        dim_max - 1
    ]  # zed is reversed to conform to the notation in paper 2022

    g_ref_faces = g_ref_faces(
        mixture, pressure, z, gravity_value, left_restriction, right_restriction
    )
    gamma_val = gamma_value(permeability)

    total_flux = [None] * mixture.num_phases

    for m in np.arange(mixture.num_phases):
        saturation_m = mixture.get_phase(m).saturation
        density_m = mixture.get_phase(m).mass_density(pressure)
        bet_faces = beta_faces(
            pressure,
            saturation_m,
            density_m,
            z,
            gravity_value,
            gamma_val,
            g_ref_faces,
            left_restriction,
            right_restriction,
            ad,
        )
        mob = mobility(saturation_m, dynamic_viscosity)

        lam_WA_faces = lambda_WA_faces(
            bet_faces, mob, left_restriction, right_restriction
        )

        delta_pot_faces = delta_potential_faces(
            pressure,
            saturation_m,
            density_m,
            z,
            gravity_value,
            left_restriction,
            right_restriction,
        )

        total_flux[m] = (
            lam_WA_faces * delta_pot_faces
        ) * transmissibility_internal_tpfa

    return total_flux


def rho_total_flux_internal(
    sd: pp.Grid,
    mixture,
    pressure,
    gravity_value,
    left_restriction,
    right_restriction,
    transmissibility_internal_tpfa,
    ad,
    dynamic_viscosity,
    dim_max,
    mobility,
    permeability,
):
    """ """
    qt = total_flux_internal(
        sd,
        mixture,
        pressure,
        gravity_value,
        left_restriction,
        right_restriction,
        transmissibility_internal_tpfa,
        ad,
        dynamic_viscosity,
        dim_max,
        mobility,
        permeability,
    )

    rho_qt = [None, None]
    for m in np.arange(mixture.num_phases):
        saturation_m = mixture.get_phase(m).saturation
        density_m = mixture.get_phase(m).mass_density(pressure)
        rho_m = hu_utils.density_internal_faces(
            saturation_m, density_m, left_restriction, right_restriction
        )

        rho_qt[m] = rho_m * qt[m]

    rho_qt = rho_qt[0] + rho_qt[1]
    return rho_qt


def rho_total_flux(
    sd: pp.Grid,
    mixture,
    pressure,
    gravity_value,
    left_restriction,
    right_restriction,
    expansion_matrix,
    transmissibility_internal_tpfa,
    ad,
    dynamic_viscosity,
    dim_max,
    mobility,
    permeability,
):
    # 0D shortcut:
    if sd.dim == 0:
        rho_qt = pp.ad.AdArray(
            np.empty((0)), sp.sparse.csr_matrix((0, pressure.jac.shape[1]))
        )
        return rho_qt

    rho_qt = expansion_matrix @ rho_total_flux_internal(
        sd,
        mixture,
        pressure,
        gravity_value,
        left_restriction,
        right_restriction,
        transmissibility_internal_tpfa,
        ad,
        dynamic_viscosity,
        dim_max,
        mobility,
        permeability,
    )

    return rho_qt
