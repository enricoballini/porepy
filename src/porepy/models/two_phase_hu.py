import scipy as sp
import numpy as np
import scipy as sp
import scipy.sparse as sps

# from types import NoneType # NoneType not present in 3.0 < python < 3.10
import porepy as pp

from typing import Callable, Optional, Type, Literal, Sequence, Union
from porepy.numerics.ad.forward_mode import AdArray, initAdArrays

import copy

import os
import sys
import pdb
import time
import warnings

os.system("clear")


def myprint(var):
    print("\n" + var + " = ", eval(var))


"""
- TODO: is it physically right to consider mu (dynamic viscosity, mu = nu/rho) constant? 
- TODO: not 100% sure about avg and int
- TODO: normalization to reduce condition number
- TODO: see todos in the code
- TODO: I'd like to reactivate the complex step to see how much worse/better it is
- TODO: fix the timestep, the simulation ends after final time
- TODO: params is a mess, it contains keys about physics and numerics used by model and newton solver. Can't I write everything directly into the model?
- TODO: the scaling how it is implemented now is a mess. There should be a method inside the model that takes care of the scaling. I don't like units
- TODO: major review of the interaction run_time_dependent_simulation-Newton-model
- TODO: with used defined intrinsic_permeability, normal_perm, grid aperture you get slightly different error numbers during newton iteration wrt the dictionary settings of those properties. No visible difference in the graphical result (paraview) tho, the newton tol is very low. That's scary anyhow, those are error only due to the implementaiton 
- TODO: rediscretization of only nonlinear terms produces slighlty different results. Is it a bug? Have I forgotten a term? 

- TODO SOON: there is something wrong with the scaling, Ka_0 shouldnt affects the results

NOTE:
- two strategies are implemented to ensure S in [0,1]: cplipping and timestep chopping
"""


class VariableDt(pp.ad.operator_functions.AbstractFunction):
    """
    - TODO: why have I had to write this operator? I must be doing something wrong...
    """

    def __init__(
        self,
        equation_system: pp.EquationSystem,
        subdomains: list[pp.Grid],
        time_manager: pp.TimeManager,
        name: str,
        array_compatible: bool = True,
    ):
        super().__init__(None, name, array_compatible)
        self._operation = pp.ad.operators.Operator.Operations.evaluate
        self.ad_compatible = False
        self.equation_system = equation_system
        self.subdomains = subdomains
        self.time_manager = time_manager
        self.result_list = []

    def get_values(self, *args_not_used: AdArray) -> np.ndarray:
        """ """
        dt = self.time_manager.dt
        dt_list = []

        for sd in self.subdomains:
            dt_list.append(dt * np.ones(sd.num_cells))  # hardcoded for two phase flow
        return np.concatenate(dt_list)

    def get_jacobian(self, *args_not_used: AdArray) -> np.ndarray:
        """ """
        jac_list = []

        for sd in self.subdomains:
            jac_list.append(
                sp.sparse.csr_matrix((sd.num_cells, self.equation_system.num_dofs()))
            )  # hardcoded for two phase flow

        return sp.sparse.vstack(jac_list)


class FunctionTotalFlux(pp.ad.operator_functions.AbstractFunction):
    """ """

    def __init__(
        self,
        func: Callable,
        mixture: pp.Mixture,
        mdg: pp.MixedDimensionalGrid,
        name: str,
        array_compatible: bool = True,
    ):
        super().__init__(func, name, array_compatible)
        self._operation = pp.ad.operators.Operator.Operations.evaluate
        self.ad_compatible = False
        self.mixture = mixture
        self.mdg = mdg

        self.result_list = []

    def get_args_from_sd_data_dictionary(self, data):
        """ """
        pressure = data["for_hu"]["pressure_AdArray"]
        gravity_value = data["for_hu"]["gravity_value"]

        left_restriction = data["for_hu"]["left_restriction"]
        right_restriction = data["for_hu"]["right_restriction"]
        expansion_matrix = data["for_hu"]["expansion_matrix"]
        transmissibility_internal_tpfa = data["for_hu"][
            "transmissibility_internal_tpfa"
        ]
        ad = True  # Sorry
        dynamic_viscosity = data["for_hu"]["dynamic_viscosity"]  # Sorry
        dim_max = data["for_hu"]["dim_max"]
        mobility = data["for_hu"]["mobility"]  # sorry
        relative_permeability = data["for_hu"]["relative_permeability"]  # sorry

        return [
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
            relative_permeability,
        ]

    def get_values(self, *args_not_used: AdArray) -> np.ndarray:
        """
        remember that *args_not_used are result[1] in _parse_operator
        - I rely on the fact the _parse first calld get_values and THEN get_jacobian
        """
        self.result_list = []  # otherwise it will remember old values...

        known_term_list = (
            []
        )  # TODO: find a better name, which is not rhs because it is not yet

        for subdomain, data in self.mdg.subdomains(return_data=True):
            args = [subdomain]

            args.append(self.mixture.mixture_for_subdomain(subdomain))  # sorry

            args_from_dict = self.get_args_from_sd_data_dictionary(data)
            args = np.hstack((args, args_from_dict))

            result = self.func(*args)
            self.result_list.append(result)

            known_term_list.append(
                result.val
            )  # equation_system.assemble_subsystem will add the "-", so it will actually become a rhs

        return np.concatenate(
            known_term_list
        )  # same as before, I want to use exacly the same commands int equation_system.assemble_subsystem

    def get_jacobian(self, *args_not_used: AdArray) -> np.ndarray:
        """
        - I rely on the fact the _parse first calls get_values and THEN get_jacobian
        """
        jac_list = []

        for result in self.result_list:
            jac_list.append(result.jac)

        return sp.sparse.vstack(jac_list)


class FunctionRhoV(pp.ad.operator_functions.AbstractFunction):
    """ """

    def __init__(
        self,
        func: Callable,
        mixture,
        mdg,
        name: str,
        array_compatible: bool = True,
    ):
        super().__init__(func, name, array_compatible)
        self._operation = pp.ad.operators.Operator.Operations.evaluate
        self.ad_compatible = False

        self.mixture = mixture
        self.mdg = mdg

        self.result_list = []

    def get_args_from_sd_data_dictionary(self, data):
        """ """

        ell = data["for_hu"]["ell"]
        pressure = data["for_hu"]["pressure_AdArray"]
        total_flux_internal = data["for_hu"]["total_flux_internal"]
        left_restriction = data["for_hu"]["left_restriction"]
        right_restriction = data["for_hu"]["right_restriction"]
        ad = True  # Sorry
        dynamic_viscosity = data["for_hu"]["dynamic_viscosity"]  # Sorry
        mobility = data["for_hu"]["mobility"]

        return [
            ell,
            pressure,
            total_flux_internal,
            left_restriction,
            right_restriction,
            ad,
            dynamic_viscosity,
            mobility,
        ]

    def get_values(self, *args_not_used: AdArray) -> np.ndarray:
        """ """
        self.result_list = []

        known_term_list = []
        for subdomain, data in self.mdg.subdomains(return_data=True):
            args = [subdomain]
            args.append(self.mixture.mixture_for_subdomain(subdomain))  # sorry
            args_from_dict = self.get_args_from_sd_data_dictionary(data)
            args = np.hstack((args, args_from_dict))

            results = self.func(*args)
            self.result_list.append(results)
            known_term_list.append(results.val)

        return np.concatenate(known_term_list)

    def get_jacobian(self, *args_not_used: AdArray) -> np.ndarray:
        """ """
        jac_list = []

        for result in self.result_list:
            jac_list.append(result.jac)

        return sp.sparse.vstack(jac_list)


class FunctionRhoG(pp.ad.operator_functions.AbstractFunction):
    """ """

    def __init__(
        self,
        func: Callable,
        mixture,
        mdg,
        name: str,
        array_compatible: bool = True,
    ):
        super().__init__(func, name, array_compatible)
        self._operation = pp.ad.operators.Operator.Operations.evaluate
        self.ad_compatible = False

        self.mixture = mixture
        self.mdg = mdg

        self.result_list = []

    def get_args_from_sd_data_dictionary(self, data):
        """ """

        ell = data["for_hu"]["ell"]
        pressure = data["for_hu"]["pressure_AdArray"]
        gravity_value = data["for_hu"]["gravity_value"]
        left_restriction = data["for_hu"]["left_restriction"]
        right_restriction = data["for_hu"]["right_restriction"]
        transmissibility_internal_tpfa = data["for_hu"][
            "transmissibility_internal_tpfa"
        ]
        ad = True  # Sorry
        dynamic_viscosity = data["for_hu"]["dynamic_viscosity"]  # Sorry
        dim_max = data["for_hu"]["dim_max"]
        mobility = data["for_hu"]["mobility"]

        return [
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
        ]

    def get_values(self, *args_not_used: AdArray) -> np.ndarray:
        """ """
        self.result_list = []

        known_term_list = []
        for subdomain, data in self.mdg.subdomains(return_data=True):
            args = [subdomain]
            args.append(self.mixture.mixture_for_subdomain(subdomain))
            args_from_dict = self.get_args_from_sd_data_dictionary(data)
            args = np.hstack((args, args_from_dict))

            result = self.func(*args)
            self.result_list.append(result)

            known_term_list.append(result.val)

        return np.concatenate(known_term_list)

    def get_jacobian(self, *args_not_used: AdArray) -> np.ndarray:
        """ """
        jac_list = []

        for result in self.result_list:
            jac_list.append(result.jac)

        return sp.sparse.vstack(jac_list)


class PrimaryVariables(pp.VariableMixin):
    """ """

    def create_variables(self) -> None:
        """ """
        self.equation_system.create_variables(
            self.pressure_variable,
            subdomains=self.mdg.subdomains(),
            tags={"si_units": "Pa"},
        )

        self.equation_system.create_variables(
            self.saturation_variable,
            subdomains=self.mdg.subdomains(),
            tags={"si_units": ""},
        )

        self.equation_system.create_variables(
            self.interface_mortar_flux_phase_0_variable,
            interfaces=self.mdg.interfaces(codim=1),
            tags={"si_units": f"m^{self.nd} * Pa"},
        )

        self.equation_system.create_variables(
            self.interface_mortar_flux_phase_1_variable,
            interfaces=self.mdg.interfaces(codim=1),
            tags={"si_units": f"m^{self.nd} * Pa"},
        )

    def pressure(self, subdomains: list[pp.Grid]) -> pp.ad.MixedDimensionalVariable:
        p = self.equation_system.md_variable(self.pressure_variable, subdomains)
        return p

    # saturation is inside mixture

    def interface_mortar_flux_phase_0(
        self, interfaces: list[pp.MortarGrid]
    ) -> pp.ad.MixedDimensionalVariable:
        """ """
        mortar_flux = self.equation_system.md_variable(
            self.interface_mortar_flux_phase_0_variable, interfaces
        )
        return mortar_flux

    def interface_mortar_flux_phase_1(
        self, interfaces: list[pp.MortarGrid]
    ) -> pp.ad.MixedDimensionalVariable:
        """ """
        mortar_flux = self.equation_system.md_variable(
            self.interface_mortar_flux_phase_1_variable, interfaces
        )
        return mortar_flux


class Equations(pp.BalanceEquation):
    """I dont see the point of splitting this class more than this, I prefer commented titles instead of mixin classes"""

    def set_equations(self) -> None:
        # pressure equation:
        subdomains = self.mdg.subdomains()
        subdomain_p_eq = self.pressure_equation(subdomains)
        self.equation_system.set_equation(subdomain_p_eq, subdomains, {"cells": 1})

        # mass balance:
        subdomains = self.mdg.subdomains()
        subdomain_mass_bal = self.mass_balance_equation(subdomains)
        self.equation_system.set_equation(subdomain_mass_bal, subdomains, {"cells": 1})

        if (
            self.mdg.num_subdomains() > 1
        ):  # pp, why do try to set morart eq even though there arent fracts?
            # mortar fluxes pahse 0:
            codim_1_interfaces = self.mdg.interfaces(codim=1)
            interface_mortar_eq_phase_0 = self.interface_mortar_flux_equation_phase_0(
                codim_1_interfaces
            )
            self.equation_system.set_equation(
                interface_mortar_eq_phase_0, codim_1_interfaces, {"cells": 1}
            )

            # mortar fluxes phase 1:
            interface_mortar_eq_phase_1 = self.interface_mortar_flux_equation_phase_1(
                codim_1_interfaces
            )
            self.equation_system.set_equation(
                interface_mortar_eq_phase_1, codim_1_interfaces, {"cells": 1}
            )

    def balance_equation_variable_dt(
        self,
        subdomains: list[pp.Grid],
        accumulation: pp.ad.Operator,
        surface_term: pp.ad.Operator,
        source: pp.ad.Operator,
        dim: int,
    ) -> pp.ad.Operator:
        """
        - TODO: why have I had to write this operator? I must be doing something wrong...
        """

        variable_dt = VariableDt(
            self.equation_system, subdomains, self.time_manager, name="variable_dt"
        )
        fake_input = self.pressure(subdomains)

        derivative_accumulation = (
            accumulation - accumulation.previous_timestep()
        ) / variable_dt(fake_input)

        div = pp.ad.Divergence(subdomains, dim=dim)

        return derivative_accumulation + div @ surface_term - source

    # PRESSURE EQUATION: -------------------------------------------------------------------------------------------------

    def eq_fcn_pressure(self, subdomains: list[pp.Grid]) -> tuple[pp.ad.Operator]:
        # accumulation term: ------------------------------
        mass_density_phase_0 = self.mixture.get_phase(0).mass_density_operator(
            subdomains, self.pressure
        ) * self.mixture.get_phase(0).saturation_operator(subdomains)

        mass_density_phase_1 = self.mixture.get_phase(1).mass_density_operator(
            subdomains, self.pressure
        ) * self.mixture.get_phase(1).saturation_operator(subdomains)

        mass_density = self.porosity(subdomains) * (
            mass_density_phase_0 + mass_density_phase_1
        )

        fake_input = self.pressure(subdomains)

        accumulation = self.volume_integral(mass_density, subdomains, dim=1)
        accumulation.set_name("accumulation_p_eq")

        # subdoamins flux: ---------------------------------------------------------
        rho_total_flux_operator = FunctionTotalFlux(
            pp.rho_total_flux,
            self.mixture,
            self.mdg,
            name="rho qt",
        )

        flux_tot = rho_total_flux_operator(fake_input)

        # interfaces flux contribution (copied from mass bal): ------------------------------------
        interfaces = self.subdomains_to_interfaces(subdomains, [1])
        mortar_projection = pp.ad.MortarProjections(
            self.mdg, subdomains, interfaces, dim=1
        )

        discr = self.ppu_discretization(
            subdomains, "darcy_flux_phase_0"
        )  # I need only bound_transport_neu, it doesn't depend on the phases, so I use the same
        neu_boundary_matrix = discr.bound_transport_neu

        flux_intf_phase_0 = (
            neu_boundary_matrix
            @ mortar_projection.mortar_to_primary_int
            @ self.interface_fluid_mass_flux_phase_0(
                self.interface_mortar_flux_phase_0(interfaces),
                interfaces,
                self.mixture.get_phase(0),
                "interface_mortar_flux_phase_0",
            )
        )

        flux_intf_phase_1 = (
            neu_boundary_matrix
            @ mortar_projection.mortar_to_primary_int
            @ self.interface_fluid_mass_flux_phase_1(
                self.interface_mortar_flux_phase_1(interfaces),
                interfaces,
                self.mixture.get_phase(1),
                "interface_mortar_flux_phase_1",
            )
        )

        flux = (
            flux_tot
            - flux_intf_phase_0
            - flux_intf_phase_1
            - neu_boundary_matrix @ self.bc_neu_phase_0(subdomains)
            - neu_boundary_matrix @ self.bc_neu_phase_1(subdomains)
        )

        # sources: --------------------------------------------------------------
        projection = pp.ad.MortarProjections(self.mdg, subdomains, interfaces)
        source_phase_0 = (
            projection.mortar_to_secondary_int
            @ self.interface_fluid_mass_flux_phase_0(
                self.interface_mortar_flux_phase_0(interfaces),
                interfaces,
                self.mixture.get_phase(0),
                "interface_mortar_flux_phase_0",
            )
        )
        source_phase_0.set_name("interface_fluid_mass_flux_source_phase_0")

        projection = pp.ad.MortarProjections(self.mdg, subdomains, interfaces)
        source_phase_1 = (
            projection.mortar_to_secondary_int
            @ self.interface_fluid_mass_flux_phase_1(
                self.interface_mortar_flux_phase_1(interfaces),
                interfaces,
                self.mixture.get_phase(1),
                "interface_mortar_flux_phase_1",
            )
        )
        source_phase_1.set_name("interface_fluid_mass_flux_source_phase_1")

        source = source_phase_0 + source_phase_1

        eq = self.balance_equation_variable_dt(
            subdomains, accumulation, flux, source, dim=1
        )
        eq.set_name("pressure_equation")

        return (
            eq,
            accumulation,
            flux_tot,
            flux_intf_phase_0,
            flux_intf_phase_1,
            flux,
            source_phase_0,
            source_phase_1,
            source,
        )

    def pressure_equation(self, subdomains: list[pp.Grid]) -> pp.ad.Operator:
        eq, _, _, _, _, _, _, _, _ = self.eq_fcn_pressure(subdomains)
        return eq

    # MASS BALANCE: ----------------------------------------------------------------------------------------------------------------

    def eq_fcn_mass(self, subdomains: list[pp.Grid]) -> tuple[pp.ad.Operator]:
        # accumulation term: ------------------------------------------------
        mass_density = (
            self.porosity(subdomains)
            * self.mixture.get_phase(self.ell).mass_density_operator(
                subdomains, self.pressure
            )
            * self.mixture.get_phase(self.ell).saturation_operator(subdomains)
        )

        fake_input = self.pressure(subdomains)

        accumulation = self.volume_integral(mass_density, subdomains, dim=1)
        accumulation.set_name("accumulation_mass_eq")

        # subdomains flux contribution: -------------------------------------
        rho_V_operator = FunctionRhoV(
            pp.rho_flux_V,
            self.mixture,
            self.mdg,
            name="rho V",
        )
        rho_G_operator = FunctionRhoG(
            pp.rho_flux_G,
            self.mixture,
            self.mdg,
            name="rho G",
        )

        flux_V_G = rho_V_operator(fake_input) + rho_G_operator(fake_input)

        # interfaces flux contribution: ------------------------------------
        interfaces = self.subdomains_to_interfaces(subdomains, [1])
        mortar_projection = pp.ad.MortarProjections(
            self.mdg, subdomains, interfaces, dim=1
        )

        discr = self.ppu_discretization(
            subdomains, "darcy_flux_phase_0"
        )  # I need only bound_transport_neu, it doesn't depend on the phases, so I use the same
        neu_boundary_matrix = discr.bound_transport_neu
        flux_intf_phase_0 = (
            neu_boundary_matrix
            @ mortar_projection.mortar_to_primary_int
            @ self.interface_fluid_mass_flux_phase_0(
                self.interface_mortar_flux_phase_0(interfaces),
                interfaces,
                self.mixture.get_phase(0),
                "interface_mortar_flux_phase_0",
            )
        )

        flux_intf_phase_1 = (
            neu_boundary_matrix
            @ mortar_projection.mortar_to_primary_int
            @ self.interface_fluid_mass_flux_phase_1(
                self.interface_mortar_flux_phase_1(interfaces),
                interfaces,
                self.mixture.get_phase(1),
                "interface_mortar_flux_phase_1",
            )
        )

        if self.ell == 0:
            flux = (
                flux_V_G
                - flux_intf_phase_0
                - neu_boundary_matrix @ self.bc_neu_phase_0(subdomains)
            )

        else:  # self.ell == 1
            flux = (
                flux_V_G
                - flux_intf_phase_1
                - neu_boundary_matrix @ self.bc_neu_phase_1(subdomains)
            )

        # sources: ---------------------------------------
        projection = pp.ad.MortarProjections(self.mdg, subdomains, interfaces)
        source_phase_0 = (
            projection.mortar_to_secondary_int
            @ self.interface_fluid_mass_flux_phase_0(
                self.interface_mortar_flux_phase_0(interfaces),
                interfaces,
                self.mixture.get_phase(0),
                "interface_mortar_flux_phase_0",
            )
        )

        projection = pp.ad.MortarProjections(self.mdg, subdomains, interfaces)
        source_phase_1 = (
            projection.mortar_to_secondary_int
            @ self.interface_fluid_mass_flux_phase_1(
                self.interface_mortar_flux_phase_1(interfaces),
                interfaces,
                self.mixture.get_phase(1),
                "interface_mortar_flux_phase_1",
            )
        )

        if self.ell == 0:
            source = source_phase_0  # I need this for the tests
            source.set_name("interface_fluid_mass_flux_source_phase_0")

        else:  # self.ell == 1:
            source = source_phase_1
            source.set_name("interface_fluid_mass_flux_source_phase_1")

        eq = self.balance_equation_variable_dt(
            subdomains, accumulation, flux, source, dim=1
        )
        eq.set_name("mass_balance_equation")

        return (
            eq,
            accumulation,
            rho_V_operator(fake_input),
            rho_G_operator(fake_input),
            flux_V_G,
            flux_intf_phase_0,
            flux_intf_phase_1,
            source_phase_0,
            source_phase_1,
        )

    def mass_balance_equation(self, subdomains: list[pp.Grid]) -> pp.ad.Operator:
        eq, _, _, _, _, _, _, _, _ = self.eq_fcn_mass(subdomains)

        return eq

    # DARCY LAWS: ----------------------------------------------------------------------------------------------------------

    def eq_fcn_mortar_phase_0(
        self, interfaces: list[pp.MortarGrid]
    ) -> tuple[pp.ad.Operator]:
        """definition of mortar flux
        - mortar flux is integrated mortar flux
        - mobility, Kr(s)/mu, is not included into this definition of mortar flux
        """
        subdomains = self.interfaces_to_subdomains(interfaces)

        projection = pp.ad.MortarProjections(self.mdg, subdomains, interfaces, dim=1)

        cell_volumes = self.wrap_grid_attribute(
            interfaces, "cell_volumes", dim=1
        )  # surface in 2D, length in 1D, the thickness isn't included. Specific_volume adds it

        trace = pp.ad.Trace(subdomains, dim=1)

        normal_gradient = pp.ad.Scalar(2) * (
            projection.secondary_to_mortar_avg
            @ self.aperture(subdomains) ** pp.ad.Scalar(-1)
        )
        normal_gradient.set_name("normal_gradient")

        pressure_l = projection.secondary_to_mortar_avg @ self.pressure(subdomains)
        pressure_h = projection.primary_to_mortar_avg @ self.pressure_trace(subdomains)
        specific_volume_intf = (
            projection.primary_to_mortar_avg
            @ trace.trace
            @ self.specific_volume(subdomains)
        )
        specific_volume_intf.set_name("specific_volume_at_interfaces")

        discr = self.interface_ppu_discretization(
            interfaces, "interface_mortar_flux_phase_0"
        )

        density_upwinded = self.var_upwinded_interfaces(
            interfaces,
            self.mixture.get_phase(0).mass_density_operator(subdomains, self.pressure),
            discr,
        )

        kn = self.normal_perm(interfaces) * normal_gradient

        delta_p = pressure_h - pressure_l
        g_term = self.interface_vector_source(interfaces)

        eq = self.interface_mortar_flux_phase_0(interfaces) - (
            cell_volumes
            * (
                kn
                * specific_volume_intf
                * (delta_p - density_upwinded / normal_gradient * g_term)
            )
        )

        eq.set_name("interface_darcy_flux_equation_phase_0")
        return (eq, kn, pressure_h, pressure_l, density_upwinded, g_term)

    def interface_mortar_flux_equation_phase_0(
        self, interfaces: list[pp.MortarGrid]
    ) -> tuple[pp.ad.Operator]:
        eq, _, _, _, _, _ = self.eq_fcn_mortar_phase_0(interfaces)
        return eq

    def interface_mortar_flux_equation_phase_1(
        self, interfaces: list[pp.MortarGrid]
    ) -> pp.ad.Operator:
        """ """
        subdomains = self.interfaces_to_subdomains(interfaces)

        projection = pp.ad.MortarProjections(self.mdg, subdomains, interfaces, dim=1)

        cell_volumes = self.wrap_grid_attribute(interfaces, "cell_volumes", dim=1)
        trace = pp.ad.Trace(subdomains, dim=1)

        normal_gradient = pp.ad.Scalar(2) * (
            projection.secondary_to_mortar_avg
            @ self.aperture(subdomains) ** pp.ad.Scalar(-1)
        )
        normal_gradient.set_name("normal_gradient")

        pressure_l = projection.secondary_to_mortar_avg @ self.pressure(subdomains)
        pressure_h = projection.primary_to_mortar_avg @ self.pressure_trace(subdomains)
        specific_volume_intf = (
            projection.primary_to_mortar_avg
            @ trace.trace
            @ self.specific_volume(subdomains)
        )
        specific_volume_intf.set_name("specific_volume_at_interfaces")

        discr = self.interface_ppu_discretization(
            interfaces, "interface_mortar_flux_phase_1"
        )

        density_upwinded = self.var_upwinded_interfaces(
            interfaces,
            self.mixture.get_phase(1).mass_density_operator(subdomains, self.pressure),
            discr,
        )

        kn = self.normal_perm(interfaces) * normal_gradient

        eq = self.interface_mortar_flux_phase_1(interfaces) - (
            cell_volumes
            * (
                kn
                * specific_volume_intf
                * (
                    pressure_h
                    - pressure_l
                    - density_upwinded
                    / normal_gradient
                    * self.interface_vector_source(interfaces)
                )
            )
        )

        eq.set_name("interface_mortar_flux_equation_phase_1")
        return eq

    # MISCELLANEA FOR EQUATIONS: -------------------------------------------------------------------------------------------------

    def interface_fluid_mass_flux_phase_0(
        self,
        mortar: pp.ad.Operator,
        interfaces: list[pp.MortarGrid],
        phase: pp.Phase,
        flux_array_key: str,
    ) -> pp.ad.Operator:
        """TODO: redundant input
        - it's stupid to have mortar as input, I added it for it for more flexibility for the tests
        """
        subdomains = self.interfaces_to_subdomains(interfaces)
        discr = self.interface_ppu_discretization(interfaces, flux_array_key)
        rho_mob = phase.mass_density_operator(
            subdomains, self.pressure
        ) * self.mobility_operator(
            subdomains, phase.saturation_operator, self.dynamic_viscosity
        )

        rho_mob_upwinded = self.var_upwinded_interfaces(
            interfaces,
            rho_mob,
            discr,
        )

        flux = mortar * rho_mob_upwinded

        flux.set_name("interface_fluid_mass_flux_phase_0")
        return flux

    def interface_fluid_mass_flux_phase_1(
        self, mortar, interfaces: list[pp.MortarGrid], phase, flux_array_key
    ) -> pp.ad.Operator:
        """ """
        subdomains = self.interfaces_to_subdomains(interfaces)
        discr = self.interface_ppu_discretization(interfaces, flux_array_key)
        rho_mob = phase.mass_density_operator(
            subdomains, self.pressure
        ) * self.mobility_operator(
            subdomains, phase.saturation_operator, self.dynamic_viscosity
        )

        rho_mob_upwinded = self.var_upwinded_interfaces(
            interfaces,
            rho_mob,
            discr,
        )
        flux = mortar * rho_mob_upwinded

        flux.set_name("interface_fluid_mass_flux_phase_1")
        return flux

    def ppu_discretization(
        self, subdomains: list[pp.Grid], flux_array_key: str
    ) -> pp.ad.UpwindAd:
        """
        flux_array_key =  either darcy_flux_phase_0 or darcy_flux_phase_1
        wrong function because it uses the same keyword for the two phases. But from Upwind discr I need only bc matrix that is phase independent, so it's ok
        """
        return pp.ad.UpwindAd(self.ppu_keyword, subdomains, flux_array_key)

    def interface_ppu_discretization(
        self, interfaces: list[pp.MortarGrid], flux_array_key: str
    ) -> pp.ad.UpwindCouplingAd:
        """
        flux_array_key = either interface_mortar_flux_phase_0 or interface_mortar_flux_phase_1
        """
        return pp.ad.UpwindCouplingAd(
            self.ppu_keyword + "_" + flux_array_key, interfaces, flux_array_key
        )

    def var_upwinded_interfaces(
        self,
        interfaces: list[pp.MortarGrid],
        advected_entity: pp.ad.Operator,
        discr: pp.ad.UpwindCouplingAd,
    ) -> pp.ad.Operator:
        """ """
        subdomains = self.interfaces_to_subdomains(interfaces)
        mortar_projection = pp.ad.MortarProjections(
            self.mdg, subdomains, interfaces, dim=1
        )

        trace = pp.ad.Trace(subdomains)

        var_upwinded: pp.ad.Operator = (
            discr.upwind_primary
            @ mortar_projection.primary_to_mortar_avg
            @ trace.trace
            @ advected_entity
            + discr.upwind_secondary
            @ mortar_projection.secondary_to_mortar_avg
            @ advected_entity
        )

        return var_upwinded

    def eq_fcn_pressure_trace(self, subdomains: list[pp.Grid]):
        """
        ref F17 f3 ->-> ... and/or paper in preparation
        """

        interfaces: list[pp.MortarGrid] = self.subdomains_to_interfaces(subdomains, [1])
        projection = pp.ad.MortarProjections(self.mdg, subdomains, interfaces, dim=1)
        discr: Union[pp.ad.TpfaAd, pp.ad.MpfaAd] = self.darcy_flux_discretization(
            subdomains
        )

        pressure_cell = discr.bound_pressure_cell @ self.pressure(subdomains)

        mass_density_phase_0 = self.mixture.get_phase(0).mass_density_operator(
            subdomains, self.pressure
        )
        mass_density_phase_1 = self.mixture.get_phase(1).mass_density_operator(
            subdomains, self.pressure
        )

        rho_mob_phase_0 = mass_density_phase_0 * self.mobility_operator(
            subdomains,
            self.mixture.get_phase(0).saturation_operator,
            self.dynamic_viscosity,
        )
        rho_mob_rho_phase_0 = (
            mass_density_phase_0
            * self.mobility_operator(
                subdomains,
                self.mixture.get_phase(0).saturation_operator,
                self.dynamic_viscosity,
            )
            * mass_density_phase_0
        )
        rho_mob_phase_1 = mass_density_phase_1 * self.mobility_operator(
            subdomains,
            self.mixture.get_phase(1).saturation_operator,
            self.dynamic_viscosity,
        )
        rho_mob_rho_phase_1 = (
            mass_density_phase_1
            * self.mobility_operator(
                subdomains,
                self.mixture.get_phase(1).saturation_operator,
                self.dynamic_viscosity,
            )
            * mass_density_phase_1
        )

        discr_up_0 = self.interface_ppu_discretization(
            interfaces, "interface_mortar_flux_phase_0"
        )
        rho_mob_upwinded_phase_0 = self.var_upwinded_interfaces(
            interfaces, rho_mob_phase_0, discr_up_0
        )
        rho_mob_rho_upwinded_phase_0 = self.var_upwinded_interfaces(
            interfaces, rho_mob_rho_phase_0, discr_up_0
        )

        discr_up_1 = self.interface_ppu_discretization(
            interfaces, "interface_mortar_flux_phase_1"
        )
        rho_mob_upwinded_phase_1 = self.var_upwinded_interfaces(
            interfaces, rho_mob_phase_1, discr_up_1
        )
        rho_mob_rho_upwinded_phase_1 = self.var_upwinded_interfaces(
            interfaces, rho_mob_rho_phase_1, discr_up_1
        )

        f_0 = self.interface_fluid_mass_flux_phase_0(
            self.interface_mortar_flux_phase_0(interfaces),
            interfaces,
            self.mixture.get_phase(0),
            "interface_mortar_flux_phase_0",
        )

        f_1 = self.interface_fluid_mass_flux_phase_1(
            self.interface_mortar_flux_phase_1(interfaces),
            interfaces,
            self.mixture.get_phase(1),
            "interface_mortar_flux_phase_1",
        )

        correction = (
            discr.bound_pressure_face
            @ (
                projection.mortar_to_primary_avg
                @ ((f_0 + f_1) / (rho_mob_upwinded_phase_0 + rho_mob_upwinded_phase_1))
            )
            - discr.bound_pressure_face
            @ (
                projection.mortar_to_primary_avg
                @ (
                    (
                        (rho_mob_rho_upwinded_phase_0 + rho_mob_rho_upwinded_phase_1)
                        / (rho_mob_upwinded_phase_0 + rho_mob_upwinded_phase_1)
                    )
                )
                * (discr.vector_source @ self.vector_source(subdomains))
            )
            # - projection.mortar_to_primary_avg @ ( ( ( rho_mob_rho_upwinded_phase_0 + rho_mob_rho_upwinded_phase_1 ) / ( rho_mob_upwinded_phase_0 + rho_mob_upwinded_phase_1 ) ) ) * ( XXX @ self.vector_source(subdomains_vector_source) )  # third version, ref F19f2, subdomains_vector_source are fractures domains, XXX
        )

        pressure_trace = pressure_cell + correction
        return pressure_trace, pressure_cell, correction

    def pressure_trace(self, subdomains: list[pp.Grid]) -> pp.ad.Operator:
        """ """
        pressure_trace, _, _ = self.eq_fcn_pressure_trace(subdomains)
        return pressure_trace

    def darcy_flux_phase_0(self, subdomains: list[pp.Grid]) -> pp.ad.Operator:
        """
        useless, need it only to make pp happy. It is used only in ppu_discretization from which I need only the Neumann boundary matrix, which doesnt depend on any flow direction...
        """

        # fake Darcy flux, only to make pp happy
        num_faces_tot = 0
        for sd in subdomains:
            num_faces_tot += sd.num_faces

        flux = pp.ad.DenseArray(np.nan * np.ones(num_faces_tot))
        flux.set_name("darcy_flux_phase_0")

        return flux

    def darcy_flux_phase_1(self, subdomains: list[pp.Grid]) -> pp.ad.Operator:
        """ """

        num_faces_tot = 0
        for sd in subdomains:
            num_faces_tot += sd.num_faces

        flux = pp.ad.DenseArray(np.nan * np.ones(num_faces_tot))
        flux.set_name("darcy_flux_phase_1")

        return flux

    def darcy_flux_discretization(self, subdomains: list[pp.Grid]) -> pp.ad.MpfaAd:
        """ """
        return pp.ad.MpfaAd(self.darcy_keyword, subdomains)

    def vector_source(
        self, subdomains: Union[list[pp.Grid], list[pp.MortarGrid]]
    ) -> pp.ad.Operator:
        """Vector source term. Represents gravity effects"""

        source = []

        # order 1. [sd_1_[x,y,z], sd_2_[x,y,z], ...] # WRONG
        # order 2. [all x, all y, all z] # WRONG
        # order 3. x y z cell1, x y z cell2, ....
        source_cell = np.zeros(self.nd)
        source_cell[self.nd - 1] = self.gravity_value

        num_cells_tot = 0
        for sd in subdomains:
            num_cells_tot += sd.num_cells  # sorry..

        source = [source_cell] * num_cells_tot

        try:
            source = pp.ad.DenseArray(np.concatenate(source))
        except:
            source = pp.ad.DenseArray(
                np.array(source)
            )  # if frac = [], interface_vector_source calls this fcn with source = [], so np.concatenate gets mad.
            # I don't like this except because if source = [] for a bug you dont see it... find a better solution
            # in pp code sometimes you see a "if len(grids) == 0: ..."
        source.set_name("vector source (gravity)")
        return source

    def interface_vector_source(
        self, interfaces: list[pp.MortarGrid]
    ) -> pp.ad.Operator:
        """ """
        normals = self.outwards_internal_boundary_normals(
            interfaces, unitary=True  # type: ignore[call-arg]
        )
        subdomain_neighbors = self.interfaces_to_subdomains(interfaces)
        projection = pp.ad.MortarProjections(
            self.mdg, subdomain_neighbors, interfaces, dim=self.nd
        )

        vector_source_all = self.vector_source(subdomain_neighbors)
        vector_source = projection.secondary_to_mortar_avg @ vector_source_all
        normals_times_source = normals * vector_source
        nd_to_scalar_sum = pp.ad.sum_operator_list(
            [e.T for e in self.basis(interfaces, dim=self.nd)]
        )
        dot_product = nd_to_scalar_sum @ normals_times_source
        dot_product.set_name("Interface vector source term")
        return dot_product

    def vector_expansion(self, subdomains: list[pp.Grid]) -> pp.ad.Operator:
        """used only for exanding the densisty in darcy fluxes.
        TODO: improve it. There should already be pp functions to do so
        """

        # I have to make the mass density a vector, from [rho_h, rho_l] to [rho_h, rho_h, rho_h, rho_l, rho_l, rho_l]
        num_cells_tot = 0
        for sd in subdomains:
            num_cells_tot += sd.num_cells

        i = 0
        offset = 0
        expansion_list = [None] * self.mdg.num_subdomains()
        for sd in subdomains:
            shape = (sd.num_cells, num_cells_tot)
            sd_expansion = sps.diags([1] * sd.num_cells, offsets=offset, shape=shape)
            sd_expansion_vect = [sd_expansion] * self.nd
            expansion_list[i] = sps.vstack(sd_expansion_vect)
            i += 1
            offset += sd.num_cells

        expansion = sps.vstack(expansion_list)
        expansion_operator = pp.ad.SparseArray(expansion)

        self.vector_expansion_operator = expansion_operator

        return expansion_operator


class ConstitutiveLawPressureMass(
    pp.constitutive_laws.DimensionReduction,
    pp.constitutive_laws.ConstantPorosity,
):
    def intrinsic_permeability(self, subdomains: list[pp.Grid]) -> pp.ad.Operator:
        """for some reason subdoamins is always a single domain"""

        sd = subdomains[0]

        if len(subdomains) > 1:
            print("\n\n\n check intrinsic_permeability\n")
            raise NotImplementedError

        if sd.dim == 3:
            permeability = pp.ad.DenseArray(1 * np.ones(sd.num_cells))
        elif sd.dim == 2:
            permeability = pp.ad.DenseArray(1 * np.ones(sd.num_cells))
        elif sd.dim == 1:
            permeability = pp.ad.DenseArray(1 * np.ones(sd.num_cells))
        else:  # 0D
            warnings.warn("setting intrinsic permeability to 0D grid")
            permeability = pp.ad.DenseArray(1 * np.ones(sd.num_cells))  # do i need it?

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
            if intf.dim == 2:
                perm[id_intf] = 1 * np.ones([intf.num_cells])
            elif intf.dim == 1:
                perm[id_intf] = 1 * np.ones([intf.num_cells])
            else:  # 0D
                perm[id_intf] = 1 * np.ones([intf.num_cells])

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
                warnings.warn("setting porosity to 0D grid")
                phi[index] = 0.25 * np.ones([sd.num_cells])

        return pp.ad.DenseArray(np.concatenate(phi))

    def grid_aperture(self, sd: pp.Grid) -> np.ndarray:
        """pay attention this is the grid aperture, not the aperture."""
        aperture = np.ones(sd.num_cells)
        residual_aperture_by_dim = [
            0.1,
            0.1,
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


class BoundaryConditionsPressureMass:
    def bc_neu_phase_0(self, subdomains: list[pp.Grid]) -> pp.ad.DenseArray:
        """ """

        bc_values: list[np.ndarray] = []

        for sd in subdomains:
            if sd.dim == self.nd:
                boundary_faces = self.domain_boundary_sides(sd).all_bf

                eps = 1e-6
                left_boundary = np.where(sd.face_centers[0] < (self.xmin + eps))[0]
                right_boundary = np.where(sd.face_centers[0] > (self.xmax - eps))[0]
                vals = np.zeros(sd.num_faces)
                vals[left_boundary] = -0.0 * sd.face_areas[left_boundary]  # inlet
                vals[right_boundary] = 0 * 2.0 * sd.face_areas[right_boundary]  # outlet
                bc_values.append(vals)

            else:
                boundary_faces = self.domain_boundary_sides(sd).all_bf

                vals = np.zeros(sd.num_faces)
                vals[boundary_faces] = (
                    0  # change this if you want different bc val # pay attention, there is a mistake in calance eq, take a look...
                )
                bc_values.append(vals)

        bc_values_array: pp.ad.DenseArray = pp.wrap_as_ad_array(
            np.hstack(bc_values), name="bc_values_flux"
        )

        return bc_values_array

    def bc_neu_phase_1(self, subdomains: list[pp.Grid]) -> pp.ad.DenseArray:
        """ """

        bc_values: list[np.ndarray] = []

        for sd in subdomains:
            if sd.dim == self.nd:
                boundary_faces = self.domain_boundary_sides(sd).all_bf

                eps = 1e-6
                left_boundary = np.where(sd.face_centers[0] < (self.xmin + eps))[0]
                right_boundary = np.where(sd.face_centers[0] > (self.xmax - eps))[0]
                vals = np.zeros(sd.num_faces)
                vals[left_boundary] = -0 * 1.0 * sd.face_areas[left_boundary]  # inlet
                vals[right_boundary] = 0.0 * sd.face_areas[right_boundary]  # outlet
                bc_values.append(vals)

            else:
                boundary_faces = self.domain_boundary_sides(sd).all_bf

                vals = np.zeros(sd.num_faces)
                vals[boundary_faces] = 0
                bc_values.append(vals)

        bc_values_array: pp.ad.DenseArray = pp.wrap_as_ad_array(
            np.hstack(bc_values), name="bc_values_flux"
        )

        return bc_values_array

    def bc_type_darcy(self, sd: pp.Grid) -> pp.BoundaryCondition:
        """required by tpfa, upwind
        TODO: bc_type_darcy is a wrong name, change it
        """

        boundary_faces = self.domain_boundary_sides(sd).all_bf

        return pp.BoundaryCondition(sd, boundary_faces, "neu")

    def bc_values_darcy(self, subdomains: list[pp.Grid]) -> pp.ad.DenseArray:
        """ """
        num_faces = sum([sd.num_faces for sd in subdomains])
        return pp.wrap_as_ad_array(0, num_faces, "bc_values_darcy")


class InitalConditionPressureMass:
    def initial_condition_common(self):
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
            saturation_values[np.where(sd.cell_centers[1] >= self.ymax / 2.5)] = (
                1.0  # TODO: HARDCODED for 2D
            )

            # if (
            #     sd.dim == 1
            # ):  # TODO: HARDCODED for 2D # thers is a is_frac attribute somewhere
            #     saturation_values = 0.8 * np.ones(sd.num_cells)

            self.equation_system.set_variable_values(
                saturation_values,
                variables=[saturation_variable],
                time_step_index=0,
                iterate_index=0,
            )

            pressure_variable = self.pressure([sd])

            pressure_values = (
                2 - 1 * sd.cell_centers[1] / self.ymax
            ) / self.p_0  # TODO: hardcoded for 2D

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


class SolutionStrategyPressureMass(pp.SolutionStrategy):
    def __init__(self, params: Optional[dict] = None) -> None:
        super().__init__(params)

        self.pressure_variable: str = "pressure"
        self.saturation_variable = "saturation"
        self.interface_mortar_flux_phase_0_variable: str = (
            "interface_mortar_flux_phase_0"
        )
        self.interface_mortar_flux_phase_1_variable: str = (
            "interface_mortar_flux_phase_1"
        )
        self.darcy_keyword: str = "flow"
        self.ppu_keyword: str = "ppu"

        self._nonlinear_discretizations = []

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

    # methods called in prepare_simulation: ------------------------------------------------------

    def clean_working_directory(self) -> None:
        """ """
        # os.system("rm " + self.output_file_name)
        # os.system("rm " + self.mass_output_file_name)
        # os.system("rm " + self.flips_file_name)
        # os.system("rm -r " + self.beta_file_name + "/*")
        print(
            "\n\n\nworking directory not cleaned --------------------------------------------\n\n\n"
        )

    def set_equation_system_manager(self) -> None:
        """ """
        if not hasattr(self, "equation_system"):
            self.equation_system = pp.ad.EquationSystem(self.mdg)

    def add_equation_system_to_phases(self) -> None:
        """ """
        for phase in self.mixture.phases:
            phase.equation_system = self.equation_system

    def set_materials(self):
        """ """
        constants = {}

        constants.update(self.params.get("material_constants", {}))

        for name, const in constants.items():
            assert isinstance(const, pp.models.material_constants.MaterialConstants)
            const.set_units(self.units)
            setattr(self, name, const)

    def reset_state_from_file(self) -> None:
        """ """
        if self.restart_options.get("restart", False):
            if self.restart_options.get("pvd_file", None) is not None:
                pvd_file = self.restart_options["pvd_file"]
                is_mdg_pvd = self.restart_options.get("is_mdg_pvd", False)
                times_file = self.restart_options.get("times_file", None)
                self.load_data_from_pvd(
                    pvd_file,
                    is_mdg_pvd,
                    times_file,
                )
            else:
                vtu_files = self.restart_options["vtu_files"]
                time_index = self.restart_options.get("time_index", -1)
                times_file = self.restart_options.get("times_file", None)
                self.load_data_from_vtu(
                    vtu_files,
                    time_index,
                    times_file,
                )
            vals = self.equation_system.get_variable_values(time_step_index=0)
            self.equation_system.set_variable_values(
                vals, iterate_index=0, time_step_index=0
            )

    def set_discretization_parameters(self) -> None:
        """ """

        for sd, data in self.mdg.subdomains(return_data=True):
            pp.initialize_data(
                sd,
                data,
                self.darcy_keyword,
                {
                    "bc": self.bc_type_darcy(sd),
                    "second_order_tensor": self.intrinsic_permeability_tensor(sd),
                    "ambient_dimension": self.nd,
                },
            )

            pp.initialize_data(
                sd,
                data,
                self.ppu_keyword,
                {
                    "bc": self.bc_type_darcy(sd),
                },
            )

        for intf, intf_data in self.mdg.interfaces(return_data=True, codim=1):
            pp.initialize_data(
                intf,
                intf_data,
                self.darcy_keyword,
                {
                    "ambient_dimension": self.nd,
                },
            )

    def discretize(self) -> None:
        """ """
        self.equation_system.discretize()

    def rediscretize(self) -> None:
        """ """
        unique_discr = pp.ad._ad_utils.uniquify_discretization_list(
            self.nonlinear_discretizations
        )
        pp.ad._ad_utils.discretize_from_list(unique_discr, self.mdg)

        # self.discretize()

    @property
    def nonlinear_discretizations(self) -> list[pp.ad._ad_utils.MergedOperator]:
        """ """
        return self._nonlinear_discretizations

    def add_nonlinear_discretization(
        self, discretization_list: list[pp.ad._ad_utils.MergedOperator]
    ) -> None:
        """ """
        for discretization in discretization_list:
            if discretization not in self._nonlinear_discretizations:
                self._nonlinear_discretizations.append(discretization)

    def set_nonlinear_discretizations(self) -> None:
        """
        only for hu. For ppu you have to upwind in sd.
        Not added here because I use ppu_discretization only for neu bc matrix, that is time (and phase) independent
        """
        # subdomains = [sd for sd in self.mdg.subdomains() if sd.dim < self.nd]
        interfaces = self.mdg.interfaces()

        # remember that ppu is used only for constan bc, so no need to rediscretize it. The rediscretizations affects interface_ppu and hu but the last is done elsewhere
        self.add_nonlinear_discretization(
            [
                self.interface_ppu_discretization(
                    interfaces, "interface_mortar_flux_phase_0"
                ).mortar_discr,  # TODO: this is a constant matrix, right? I can remove it...
                self.interface_ppu_discretization(
                    interfaces, "interface_mortar_flux_phase_0"
                ).flux,
                self.interface_ppu_discretization(
                    interfaces, "interface_mortar_flux_phase_0"
                ).upwind_primary,
                self.interface_ppu_discretization(
                    interfaces, "interface_mortar_flux_phase_0"
                ).upwind_secondary,
            ]
        )

        self.add_nonlinear_discretization(
            [
                self.interface_ppu_discretization(
                    interfaces, "interface_mortar_flux_phase_1"
                ).mortar_discr,
                self.interface_ppu_discretization(
                    interfaces, "interface_mortar_flux_phase_1"
                ).flux,
                self.interface_ppu_discretization(
                    interfaces, "interface_mortar_flux_phase_1"
                ).upwind_primary,
                self.interface_ppu_discretization(
                    interfaces, "interface_mortar_flux_phase_1"
                ).upwind_secondary,
            ]
        )

    def _initialize_linear_solver(self) -> None:
        """ """
        solver = self.params["linear_solver"]
        self.linear_solver = solver

        if solver not in ["scipy_sparse", "pypardiso", "umfpack"]:
            raise ValueError(f"Unknown linear solver {solver}")

    def computations_for_hu(self) -> None:
        """
        this will change a lot so it is useless to improve it right now
        """
        for sd, data in self.mdg.subdomains(return_data=True):
            data["for_hu"] = (
                {}
            )  # this is the first call, so i need to create the dictionary first

            data["for_hu"]["gravity_value"] = self.gravity_value
            data["for_hu"]["ell"] = self.ell
            data["for_hu"]["dynamic_viscosity"] = self.dynamic_viscosity

            if sd.dim == 0:
                data["for_hu"]["left_restriction"] = None
                data["for_hu"]["right_restriction"] = None
                data["for_hu"]["expansion_matrix"] = None
                data["for_hu"]["transmissibility_internal_tpfa"] = None
                data["for_hu"]["dim_max"] = None
                data["for_hu"]["mobility"] = None
                data["for_hu"]["relative_permeability"] = None

            else:
                (
                    left_restriction,
                    right_restriction,
                ) = pp.numerics.fv.hybrid_upwind_utils.restriction_matrices_left_right(
                    sd
                )
                data["for_hu"]["left_restriction"] = left_restriction
                data["for_hu"]["right_restriction"] = right_restriction
                data["for_hu"]["expansion_matrix"] = (
                    pp.numerics.fv.hybrid_upwind_utils.expansion_matrix(sd)
                )

                pp.numerics.fv.hybrid_upwind_utils.compute_transmissibility_tpfa(
                    sd, data, keyword="flow"
                )

                (
                    _,
                    transmissibility_internal,
                ) = pp.numerics.fv.hybrid_upwind_utils.get_transmissibility_tpfa(
                    sd, data, keyword="flow"
                )
                data["for_hu"][
                    "transmissibility_internal_tpfa"
                ] = transmissibility_internal
                data["for_hu"]["dim_max"] = self.mdg.dim_max()
                data["for_hu"]["mobility"] = self.mobility
                data["for_hu"]["relative_permeability"] = self.relative_permeability

    def before_nonlinear_iteration(self) -> None:
        """
        - the comuication with HU is entirely through the data dictionary of each subdomain
        """
        for sd, data in self.mdg.subdomains(return_data=True):
            # OLD:
            # pressure_adarray = self.pressure([sd]).evaluate(self.equation_system)
            # left_restriction = data["for_hu"]["left_restriction"]
            # right_restriction = data["for_hu"][
            #     "right_restriction"
            # ]  # created in prepare_simulation
            # transmissibility_internal_tpfa = data["for_hu"][
            #     "transmissibility_internal_tpfa"
            # ]
            # ad = True
            # dynamic_viscosity = data["for_hu"]["dynamic_viscosity"]
            # dim_max = data["for_hu"]["dim_max"]

            # t = time.time()
            total_flux_internal = (
                pp.numerics.fv.hybrid_weighted_average.total_flux_internal(
                    sd,
                    self.mixture.mixture_for_subdomain(sd),
                    self.pressure([sd]).evaluate(self.equation_system),
                    self.gravity_value,
                    data["for_hu"]["left_restriction"],
                    data["for_hu"]["right_restriction"],
                    data["for_hu"]["transmissibility_internal_tpfa"],
                    True,  # ad
                    data["for_hu"]["dynamic_viscosity"],
                    data["for_hu"]["dim_max"],
                    self.mobility,
                    self.relative_permeability,
                )
            )
            # print("time total_flux_internal = ", time.time() - t)

            data["for_hu"]["total_flux_internal"] = (
                total_flux_internal[0] + total_flux_internal[1]
            )

            data["for_hu"]["pressure_AdArray"] = self.pressure([sd]).evaluate(
                self.equation_system
            )

        for intf, data in self.mdg.interfaces(return_data=True, codim=1):
            vals = (
                self.interface_mortar_flux_phase_0([intf])
                .evaluate(self.equation_system)
                .val
            )
            data[pp.PARAMETERS][
                self.ppu_keyword + "_" + "interface_mortar_flux_phase_0"
            ].update({"interface_mortar_flux_phase_0": vals})

            vals = (
                self.interface_mortar_flux_phase_1([intf])
                .evaluate(self.equation_system)
                .val
            )
            data[pp.PARAMETERS][
                self.ppu_keyword + "_" + "interface_mortar_flux_phase_1"
            ].update({"interface_mortar_flux_phase_1": vals})

        self.set_discretization_parameters()
        self.rediscretize()

    def after_nonlinear_iteration(self, solution_vector: np.ndarray) -> None:
        """ """
        self._nonlinear_iteration += 1
        self.equation_system.shift_iterate_values()
        self.equation_system.set_variable_values(
            values=solution_vector, additive=True, iterate_index=0
        )
        self.clip_saturation()

    def clip_saturation(self) -> None:
        """TODO: very bad implementation..."""
        clipped_saturation = np.clip(
            self.equation_system.get_variable_values(
                [self.saturation_variable], iterate_index=0
            ),
            0,
            1,
        )
        self.equation_system.set_variable_values(
            clipped_saturation,
            [self.saturation_variable],
            iterate_index=0,
            additive=False,
        )

    def after_nonlinear_convergence(
        self, solution: np.ndarray, errors: float, iteration_counter: int
    ) -> None:
        """useless solution argument, that is dx of newton, not the current solution"""
        solution = self.equation_system.get_variable_values(
            iterate_index=0
        )  # this is the current solution

        self.equation_system.shift_time_step_values()
        self.equation_system.set_variable_values(
            values=solution, time_step_index=0, additive=False
        )
        self.convergence_status = True
        self.save_data_time_step()

    def after_nonlinear_failure(
        self, solution: np.ndarray, errors: float, iteration_counter: int
    ) -> None:
        """ """

        if self._is_nonlinear_problem():
            print("\n===========================================")
            print(
                "Nonlinear iterations did not converge. I'm going to reduce the time step"
            )
            print("===========================================")

            prev_sol = self.equation_system.get_variable_values(time_step_index=0)

            self.equation_system.set_variable_values(
                values=prev_sol, additive=False, iterate_index=0
            )

        else:
            raise ValueError("Tried solving singular matrix for the linear problem.")

    def write_newton_info(
        self,
        time: np.ndarray,
        time_steps: np.ndarray,
        time_chops: np.ndarray,
        cumulative_iterations: np.ndarray,
        global_cumulative_iterations: np.ndarray,
        last_iterations: np.ndarray,
    ) -> None:
        """
        time_step = dt used to reach time starting from time-dt
        cumulative_iteration_counter = cumulative inside the timestep
        """
        wasted_iterations = cumulative_iterations - last_iterations

        with open(self.output_file_name, "a") as f:
            info = np.array(
                [
                    time,
                    time_steps,
                    time_chops,
                    cumulative_iterations,
                    global_cumulative_iterations,
                    last_iterations,
                    wasted_iterations,
                ]
            )

            np.savetxt(f, info.reshape(1, -1), delimiter=",")

    def flip_flop(self, dim_to_check):
        """very inefficient, i want to make this function as independent as possible"""

        for sd, data in self.mdg.subdomains(return_data=True):
            if sd.dim == dim_to_check:  # for simplicity...
                # total flux flips:
                pressure_adarray = self.pressure([sd]).evaluate(self.equation_system)
                left_restriction = data["for_hu"]["left_restriction"]
                right_restriction = data["for_hu"]["right_restriction"]
                transmissibility_internal_tpfa = data["for_hu"][
                    "transmissibility_internal_tpfa"
                ]
                ad = True
                dynamic_viscosity = data["for_hu"]["dynamic_viscosity"]
                dim_max = data["for_hu"]["dim_max"]
                total_flux_internal = (
                    pp.numerics.fv.hybrid_weighted_average.total_flux_internal(
                        sd,
                        self.mixture.mixture_for_subdomain(sd),
                        pressure_adarray,
                        self.gravity_value,
                        left_restriction,
                        right_restriction,
                        transmissibility_internal_tpfa,
                        ad,
                        dynamic_viscosity,
                        dim_max,
                        self.mobility,
                        self.relative_permeability,
                    )
                )
                sign_total_flux_internal = np.sign(
                    total_flux_internal[0].val + total_flux_internal[1].val
                )
                if (
                    self.sign_total_flux_internal_prev is None
                ):  # first iteration after initial condition
                    self.sign_total_flux_internal_prev = sign_total_flux_internal

                number_flips_qt = np.sum(
                    np.not_equal(
                        self.sign_total_flux_internal_prev, sign_total_flux_internal
                    )
                )

                self.sign_total_flux_internal_prev = sign_total_flux_internal

                # omega_0 flips:
                z = -sd.cell_centers[dim_max - 1]

                saturation_list = [None] * self.mixture.num_phases
                g_list = [None] * self.mixture.num_phases
                mobility_list = [None] * self.mixture.num_phases

                for phase_id in np.arange(self.mixture.num_phases):
                    saturation = self.mixture.get_phase(phase_id).saturation
                    saturation_list[phase_id] = saturation
                    rho = self.mixture.get_phase(phase_id).mass_density(
                        self.pressure([sd]).evaluate(self.equation_system).val
                    )
                    rho = pp.hu_utils.density_internal_faces(
                        saturation, rho, left_restriction, right_restriction
                    )
                    g_list[phase_id] = pp.hu_utils.g_internal_faces(
                        z, rho, self.gravity_value, left_restriction, right_restriction
                    )
                    mobility_list[phase_id] = self.mobility(
                        saturation, dynamic_viscosity
                    )

                omega_0 = pp.omega(
                    self.mixture.num_phases,
                    0,
                    mobility_list,
                    g_list,
                    left_restriction,
                    right_restriction,
                    ad,
                )

                sign_omega_0 = np.sign(omega_0.val)

                if self.sign_omega_0_prev is None:
                    self.sign_omega_0_prev = sign_omega_0

                number_flips_omega_0 = np.sum(
                    np.not_equal(self.sign_omega_0_prev, sign_omega_0)
                )

                self.sign_omega_0_prev = sign_omega_0

                # omega_1 flips
                omega_1 = pp.omega(
                    self.mixture.num_phases,
                    1,
                    mobility_list,
                    g_list,
                    left_restriction,
                    right_restriction,
                    ad,
                )

                sign_omega_1 = np.sign(omega_1.val)

                if self.sign_omega_1_prev is None:
                    self.sign_omega_1_prev = sign_omega_1

                number_flips_omega_1 = np.sum(
                    np.not_equal(self.sign_omega_1_prev, sign_omega_1)
                )

                self.sign_omega_1_prev = sign_omega_1

        return np.array(
            [sign_total_flux_internal, sign_omega_0, sign_omega_1]
        ), np.array([number_flips_qt, number_flips_omega_0, number_flips_omega_1])

    def save_flip_flop(
        self,
        time: np.ndarray,
        cumulative_flips: np.ndarray,
        global_cumulative_flips: np.ndarray,
    ) -> None:
        """
        time: there is something wrong, called time_manager here
        cumulative_flip: number of flips for each upwind direction inside the timestep (cumulative bcs of the time chopping)
        global_cumulative_flips: cumulative number of flips for each upwind dir for initial time to time
        """

        with open(self.flips_file_name, "a") as f:
            np.savetxt(
                f,
                np.array(
                    [
                        self.time_manager.time,
                        *cumulative_flips,
                        *global_cumulative_flips,
                    ]
                ).reshape(1, -1),
                delimiter=",",
            )

    def beta_faces_model(self, dim_to_check):
        """as flip_flop, very inefficient implementation, I want this method to be independent"""

        out = self.mdg.subdomains(return_data=True, dim=dim_to_check)[0]
        sd = out[0]
        data = out[1]

        left_restriction = data["for_hu"]["left_restriction"]
        right_restriction = data["for_hu"]["right_restriction"]

        def gamma_value(permeability):
            """ """
            alpha = 1.0  # as in the paper 2022

            kr0 = permeability(saturation=1)

            def second_derivative(permeability, val):
                """sorry, I'm lazy..."""
                h = 1e-4
                return (
                    permeability(val + h)
                    - 2 * permeability(val)
                    + permeability(val - h)
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
            density_faces_0 = pp.hu_utils.density_internal_faces(
                mixture.get_phase(0).saturation,
                mixture.get_phase(0).mass_density(pressure),
                left_restriction,
                right_restriction,
            )
            density_faces_1 = pp.hu_utils.density_internal_faces(
                mixture.get_phase(1).saturation,
                mixture.get_phase(1).mass_density(pressure),
                left_restriction,
                right_restriction,
            )

            density_max = pp.ad.maximum(density_faces_0, density_faces_1)

            g_ref = (
                density_max
                * gravity_value
                * (left_restriction @ z - right_restriction @ z)
            )
            return g_ref

        pressure = self.pressure([sd]).evaluate(self.equation_system)
        saturation = (
            self.mixture.get_phase(0)
            .saturation_operator([sd])
            .evaluate(self.equation_system)
            .val
        )
        density = self.mixture.get_phase(0).mass_density(pressure).val
        z = -sd.cell_centers[1]  # HARDCODED for 2D
        gravity_value = self.gravity_value
        gamma_val = gamma_value(self.relative_permeability)
        g_ref = g_ref_faces(
            self.mixture,
            pressure,
            z,
            gravity_value,
            left_restriction,
            right_restriction,
        )

        c_faces_ref = 0

        density_internal_faces = pp.hu_utils.density_internal_faces(
            saturation, density, left_restriction, right_restriction
        )
        g_internal_faces = pp.hu_utils.g_internal_faces(
            z,
            density_internal_faces,
            gravity_value,
            left_restriction,
            right_restriction,
        )
        delta_pot_faces = (
            left_restriction @ pressure - right_restriction @ pressure
        ) - g_internal_faces

        tmp = gamma_val / (pp.ad.abs(g_ref) + c_faces_ref + 1e-8)

        tmp = -pp.ad.functions.maximum(-tmp, -1e6)

        beta_faces = 0.5 + 1 / np.pi * pp.ad.arctan(tmp * delta_pot_faces)
        beta_faces = beta_faces.val
        coordinates = sd.face_centers[[0, 1]][:, sd.get_internal_faces()]
        time = np.around(self.time_manager.time, decimals=6)

        with open(self.beta_file_name + "/BETA_" + str(time), "w") as f:
            info = np.concatenate(
                [
                    time * np.ones(beta_faces.shape[0]).reshape(1, -1),
                    coordinates,
                    beta_faces.reshape(1, -1),
                    delta_pot_faces.val.reshape(1, -1),
                ]
            )
            np.savetxt(f, info.T)

        with open(self.beta_file_name + "/BETA_TIME", "a") as f:
            np.savetxt(f, np.array([time]))

    def compute_mass(self):
        """ """
        subdomains = self.mdg.subdomains()
        saturation_phase_0 = self.mixture.get_phase(0).saturation_operator(subdomains)
        volumes = self.specific_volume(subdomains) * self.wrap_grid_attribute(
            subdomains, "cell_volumes", dim=1
        )
        density_phase_0 = self.mixture.get_phase(0).mass_density_operator(
            subdomains, self.pressure
        )
        phi = self.porosity(subdomains)

        mass_cells = volumes * phi * saturation_phase_0 * density_phase_0
        mass_cells_phase_0 = mass_cells.evaluate(self.equation_system).val

        mass_tot_phase_0 = np.sum(mass_cells_phase_0)

        with open(self.mass_output_file_name, "a") as f:
            info = np.array([self.time_manager.time, mass_tot_phase_0])

            np.savetxt(f, info.reshape(1, -1), delimiter=",")

    def eb_after_timestep(self):
        """used for plots, tests, and more"""

        self.compute_mass()
        self.beta_faces_model(dim_to_check=self.mdg.dim_max())


class MyModelGeometry(pp.ModelGeometry):
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

        # structired square:
        # self._domain = pp.applications.md_grids.domains.nd_cube_domain(2, self.size)

        # unstructred unit square:
        bounding_box = {
            "xmin": self.xmin,
            "xmax": self.xmax,
            "ymin": self.ymin,
            "ymax": self.ymax,
        }  # , "zmin": 0, "zmax": 1}
        self._domain = pp.Domain(bounding_box=bounding_box)

    def set_fractures(self) -> None:
        """ """
        frac1 = pp.LineFracture(np.array([[0.2, 0.8], [0.6, 0.6]]))
        frac2 = pp.LineFracture(np.array([[0.2, 0.8], [0.2, 0.2]]))
        frac3 = pp.LineFracture(np.array([[0.5, 0.5], [0.2, 0.8]]))

        frac4 = pp.LineFracture(np.array([[0.2, 0.8], [0.2, 0.8]]))

        frac5 = pp.LineFracture(np.array([[0.0, 0.8], [0.1, 0.3]]))
        frac6 = pp.LineFracture(np.array([[0.5, 0.8], [0.5, 0.8]]))

        # frac1 = pp.PlaneFracture(np.array([[0.2, 0.7, 0.7, 0.2],[0.2, 0.2, 0.8, 0.8],[0.5, 0.5, 0.5, 0.5]]))
        # self._fractures: list = [frac1, frac2, frac3, frac4, frac5]
        self._fractures = []  # [frac3, frac6]

    def meshing_arguments(self) -> dict[str, float]:
        """ """
        default_meshing_args: dict[str, float] = {
            "cell_size": 0.1 / self.L_0,
            "cell_size_fracture": 0.02 / self.L_0,
        }
        return self.params.get("meshing_arguments", default_meshing_args)


class TimeManagerPP(pp.TimeManager):
    def compute_time_step(
        self,
        is_converged,
        iterations: Optional[int] = None,
        recompute_solution: bool = False,
        threshold=1e-11,
    ):
        if self.time >= self.time_final:
            previous_dt = self.dt
            self.dt = self.time_final - self.time
            return (previous_dt, self.dt)  # wrong, it returns a negative dt

        # If the time step is constant, always return that value
        if self.is_constant:
            # Some sanity checks
            if self._iters is not None:
                msg = (
                    f"iterations '{self._iters}' has no effect if time step is "
                    "constant."
                )
                warnings.warn(msg)
            if self._recomp_sol:
                msg = "recompute_solution=True has no effect if time step is constant."
                warnings.warn(msg)

            return (self.dt, self.dt_init)

        if is_converged:
            previous_dt = self.dt
            self.dt = self.dt_min_max[1]
            return (previous_dt, self.dt)

        else:
            previous_dt = self.dt
            self.dt = self.dt / 2  # TODO: add "till dt > dt_min"
            if self.dt < threshold:
                warnings.warn("Timestep smaller than threshold")
                sys.exit()  # TODO: do something more gentle...

            return (previous_dt, self.dt)

    def decrease_time(self) -> None:
        """Decrease simulation time by the current time step."""
        self.time -= self.dt


class PartialFinalModel(
    PrimaryVariables,
    Equations,
    ConstitutiveLawPressureMass,
    BoundaryConditionsPressureMass,
    InitalConditionPressureMass,
    SolutionStrategyPressureMass,
    MyModelGeometry,
    pp.DataSavingMixin,
):
    """ """


if __name__ == "__main__":
    # Scaling:
    # there something conceptually wrong, scaling should be unrelated to units and easy to modify. Should be part of the model but with the current structure of the code it's too complex

    print("\nSCALING: ======================================")
    phi_0 = 1
    L_0 = 1
    gravity_0 = 1
    dynamic_viscosity_0 = 1
    rho_0 = 1  # |rho_phase_0-rho_phase_1|
    p_0 = 1
    Ka_0 = 1

    # Gravity number:
    u_0 = Ka_0 * p_0 / (dynamic_viscosity_0 * L_0)
    t_0 = L_0 / u_0

    gravity_number = Ka_0 * rho_0 * gravity_0 / (dynamic_viscosity_0 * u_0)

    print("u_0 = ", u_0)
    print("t_0 = ", t_0)
    print("gravity_number = ", gravity_number)
    print(
        "pay attention: gravity number is not influenced by Ka_0 and dynamic_viscosity_0"
    )

    # Eb Ar number: # checked: same Eb Ar number => same dynamics
    p_0 = dynamic_viscosity_0**2 / Ka_0 / rho_0
    u_0 = Ka_0 * p_0 / dynamic_viscosity_0 / L_0
    t_0 = phi_0 * L_0 / u_0

    eb_number = rho_0**2 * gravity_0 * L_0 * Ka_0 * phi_0 / dynamic_viscosity_0**2
    print("Eb Ar number = ", eb_number)
    print("=========================================\n")

    fluid_constants = pp.FluidConstants({})
    solid_constants = pp.SolidConstants(
        {
            "porosity": None,
            "intrinsic_permeability": None,  # / Ka_0,
            "normal_permeability": None,  # / Ka_0 # this does NOT include the aperture
            "residual_aperture": None,  # / L_0
        }
    )

    material_constants = {"fluid": fluid_constants, "solid": solid_constants}

    folder_name = "./visualization"

    time_manager = TimeManagerPP(
        schedule=np.array([0, 0.5]) / t_0,
        dt_init=5e-2 / t_0,
        dt_min_max=np.array([1e-5, 5e-2]) / t_0,
        constant_dt=False,
        iter_max=20,
        print_info=True,
        folder_name=folder_name,
    )

    params = {
        "material_constants": material_constants,
        "max_iterations": 10,
        "nl_convergence_tol": 1e-4,
        "nl_divergence_tol": 1e5,
        "time_manager": time_manager,
        "folder_name": folder_name,
    }

    wetting_phase = pp.composite.phase.Phase(
        rho0=1 / rho_0, p0=p_0, beta=1e-10
    )  # TODO: improve that p0, different menaing wrt rho0
    non_wetting_phase = pp.composite.phase.Phase(rho0=0.5 / rho_0, p0=p_0, beta=1e-10)

    mixture = pp.Mixture()
    mixture.add([wetting_phase, non_wetting_phase])

    class FinalModel(PartialFinalModel):
        def __init__(self, params: Optional[dict] = None):
            super().__init__(params)

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
            self.gravity_value = 1 / self.gravity_0  # pp.GRAVITY_ACCELERATION
            self.dynamic_viscosity = (
                1 / self.dynamic_viscosity_0
            )  # TODO: wrong place...

            self.xmin = 0 / self.L_0
            self.xmax = 1 / self.L_0
            self.ymin = 0 / self.L_0
            self.ymax = 1 / self.L_0

            self.relative_permeability = (
                pp.tobedefined.relative_permeability.rel_perm_quadratic
            )
            self.mobility = pp.tobedefined.mobility.mobility(self.relative_permeability)
            self.mobility_operator = pp.tobedefined.mobility.mobility_operator(
                self.mobility
            )

            self.number_upwind_dirs = (
                3  # used in run_models and nonlinear_solvers to initialize arrays
            )
            self.sign_total_flux_internal_prev = None
            self.sign_omega_0_prev = None
            self.sign_omega_1_prev = None

            self.output_file_name = "./OUTPUT_NEWTON_INFO_HU"
            self.mass_output_file_name = "./MASS_OVER_TIME_HU"
            self.flips_file_name = "./FLIPS_HU"
            self.beta_file_name = "./BETA"

    model = FinalModel(params)

    pp.run_time_dependent_model(model, params)
