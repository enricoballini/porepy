import numpy as np
import scipy as sp
import porepy as pp

from typing import Callable, Optional, Type, Literal, Sequence, Union
from porepy.numerics.ad.forward_mode import AdArray, initAdArrays

import test_hu_model

import os
import sys
import pdb

os.system("clear")


def myprint(var):
    print("\n" + var + " = ", eval(var))


"""
- copied from test_hu_model.py
"""


class EquationsPPU(test_hu_model.Equations):
    """I dont see the point of splitting this class more than this, I prefer commented titles instead of mixin classes"""

    # PRESSURE EQUATION: -------------------------------------------------------------------------------------------------

    def eq_fcn_pressure(self, subdomains):  # I suck in python. I need this for tests.
        # accumulation term: ------------------------------
        mass_density_phase_0 = self.mixture.get_phase(0).mass_density_operator(
            subdomains, self.pressure
        ) * self.mixture.get_phase(0).saturation_operator(subdomains)

        mass_density_phase_1 = self.mixture.get_phase(1).mass_density_operator(
            subdomains, self.pressure
        ) * self.mixture.get_phase(1).saturation_operator(subdomains)

        mass_density = self.porosity(subdomains) * (
            pp.ad.Scalar(0)*mass_density_phase_0 + mass_density_phase_1
        ) 

        accumulation = self.volume_integral(mass_density, subdomains, dim=1)
        accumulation.set_name("fluid_mass_p_eq")

        # subdomains flux: -----------------------------------------------------------
        discr = self.ppu_discretization(subdomains, "darcy_flux_phase_0")
        mob_rho = self.mixture.get_phase(0).mass_density_operator(subdomains, self.pressure) * self.mobility_operator(subdomains, self.mixture.get_phase(0).saturation_operator, self.dynamic_viscosity)

        darcy_flux_phase_0 = self.darcy_flux_phase_0(subdomains, self.mixture.get_phase(0))
        interfaces = self.subdomains_to_interfaces(subdomains, [1])
        mortar_projection = pp.ad.MortarProjections(
            self.mdg, subdomains, interfaces, dim=1
        )

        flux_phase_0: pp.ad.Operator = (
            darcy_flux_phase_0 * (discr.upwind @ mob_rho)
        )
    
        discr = self.ppu_discretization(subdomains, "darcy_flux_phase_1")
        mob_rho = self.mixture.get_phase(1).mass_density_operator(subdomains, self.pressure) * self.mobility_operator(
            subdomains, self.mixture.get_phase(1).saturation_operator, self.dynamic_viscosity)

        darcy_flux_phase_1 = self.darcy_flux_phase_1(subdomains, self.mixture.get_phase(1))
        interfaces = self.subdomains_to_interfaces(subdomains, [1])
        mortar_projection = pp.ad.MortarProjections(
            self.mdg, subdomains, interfaces, dim=1
        )

        flux_phase_1: pp.ad.Operator = (
            darcy_flux_phase_1 * (discr.upwind @ mob_rho)
        )

        flux_tot = flux_phase_0 + flux_phase_1

        # interfaces flux contribution (copied from mass bal): ------------------------------------
        interfaces = self.subdomains_to_interfaces(subdomains, [1])
        mortar_projection = pp.ad.MortarProjections(
            self.mdg, subdomains, interfaces, dim=1
        )

        discr = self.ppu_discretization(subdomains, "darcy_flux_phase_0")
        flux_intf_phase_0 = (
            discr.bound_transport_neu
            @ mortar_projection.mortar_to_primary_int
            @ self.interface_fluid_mass_flux_phase_0(
                self.interface_mortar_flux_phase_0(interfaces),
                interfaces,
                self.mixture.get_phase(0),
                "interface_mortar_flux_phase_0",
            )
        )

        discr = self.ppu_discretization(subdomains, "darcy_flux_phase_1")
        flux_intf_phase_1 = (
            discr.bound_transport_neu
            @ mortar_projection.mortar_to_primary_int
            @ self.interface_fluid_mass_flux_phase_1(
                self.interface_mortar_flux_phase_1(interfaces),
                interfaces,
                self.mixture.get_phase(1),
                "interface_mortar_flux_phase_1",
            )
        )

        discr_phase_0 = self.ppu_discretization(subdomains, "darcy_flux_phase_0")
        discr_phase_1 = self.ppu_discretization(subdomains, "darcy_flux_phase_1")
        
        flux = ( flux_tot - flux_intf_phase_0 - flux_intf_phase_1
                - discr_phase_0.bound_transport_neu @ self.bc_neu_phase_0(subdomains)
                - discr_phase_1.bound_transport_neu @ self.bc_neu_phase_1(subdomains) ) 

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

        eq = self.balance_equation(
            subdomains, accumulation, flux, source, dim=1
        )  # * pp.ad.Scalar(1e6)
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

    def eq_fcn_mass(self, subdomains):  # I suck in python. I need this for tests.
        # accumulation term: ------------------------------------------------
        mass_density = (
            self.porosity(subdomains)
            * self.mixture.get_phase(self.ell).mass_density_operator(
                subdomains, self.pressure
            )
            * self.mixture.get_phase(self.ell).saturation_operator(subdomains)
        )
        accumulation = self.volume_integral(mass_density, subdomains, dim=1)
        accumulation.set_name("fluid_mass_mass_eq")

        # subdomains flux contribution: -------------------------------------
        if self.ell == 0:
            discr = self.ppu_discretization(subdomains, "darcy_flux_phase_0")
            mob_rho = self.mixture.get_phase(0).mass_density_operator(subdomains, self.pressure) * self.mobility_operator(
                subdomains, self.mixture.get_phase(0).saturation_operator, self.dynamic_viscosity)

            darcy_flux_phase_0 = self.darcy_flux_phase_0(subdomains, self.mixture.get_phase(0))
            interfaces = self.subdomains_to_interfaces(subdomains, [1])
            mortar_projection = pp.ad.MortarProjections(
                self.mdg, subdomains, interfaces, dim=1
            )

            flux_phase_0: pp.ad.Operator = (
                darcy_flux_phase_0 * (discr.upwind @ mob_rho)
            )

            flux_no_intf = flux_phase_0 # just to keep the structure of hu code

        else: # self.ell == 1
            discr = self.ppu_discretization(subdomains, "darcy_flux_phase_1")
            mob_rho = self.mixture.get_phase(1).mass_density_operator(subdomains, self.pressure) * self.mobility_operator(
                subdomains, self.mixture.get_phase(1).saturation_operator, self.dynamic_viscosity)

            darcy_flux_phase_1 = self.darcy_flux_phase_1(subdomains, self.mixture.get_phase(1))
            interfaces = self.subdomains_to_interfaces(subdomains, [1])
            mortar_projection = pp.ad.MortarProjections(
                self.mdg, subdomains, interfaces, dim=1
            )

            flux_phase_1: pp.ad.Operator = (
                darcy_flux_phase_1 * (discr.upwind @ mob_rho)
            )

            flux_no_intf = flux_phase_1

        # interfaces flux contribution: ------------------------------------
        interfaces = self.subdomains_to_interfaces(subdomains, [1])
        mortar_projection = pp.ad.MortarProjections(
            self.mdg, subdomains, interfaces, dim=1
        )

        discr = self.ppu_discretization(subdomains, "darcy_flux_phase_0")
        flux_intf_phase_0 = (
            discr.bound_transport_neu  # -1,0,1 matrix
            @ mortar_projection.mortar_to_primary_int
            @ self.interface_fluid_mass_flux_phase_0(
                self.interface_mortar_flux_phase_0(interfaces),
                interfaces,
                self.mixture.get_phase(0),
                "interface_mortar_flux_phase_0",
            )
        )  # sorry, I need to test it

        discr = self.ppu_discretization(subdomains, "darcy_flux_phase_1")
        flux_intf_phase_1 = (
            discr.bound_transport_neu
            @ mortar_projection.mortar_to_primary_int
            @ self.interface_fluid_mass_flux_phase_1(
                self.interface_mortar_flux_phase_1(interfaces),
                interfaces,
                self.mixture.get_phase(1),
                "interface_mortar_flux_phase_1",
            )
        )

        discr_phase_0 = self.ppu_discretization(subdomains, "darcy_flux_phase_0")
        discr_phase_1 = self.ppu_discretization(subdomains, "darcy_flux_phase_1")
        
        if (
            self.ell == 0
        ):  # TODO: move the flux computation inside if (which btw could be removed) after all the bugs are fixed
            flux = flux_phase_0 - flux_intf_phase_0 - discr_phase_0.bound_transport_neu @ self.bc_neu_phase_0(subdomains)

        else:  # self.ell == 1
            flux = flux_phase_1 - flux_intf_phase_1 - discr_phase_1.bound_transport_neu @ self.bc_neu_phase_1(subdomains)

        flux.set_name("ppu_flux")

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
        )  # sorry, there is a bug and I need to output everything for the tests

        projection = pp.ad.MortarProjections(self.mdg, subdomains, interfaces)
        source_phase_1 = (
            projection.mortar_to_secondary_int
            @ self.interface_fluid_mass_flux_phase_1(
                self.interface_mortar_flux_phase_1(interfaces),
                interfaces,
                self.mixture.get_phase(1),
                "interface_mortar_flux_phase_1",
            )
        )  # sorry, there is a bug and I need to output everything for the tests

        if self.ell == 0:
            source = source_phase_0  # I need this for the tests
            source.set_name("interface_fluid_mass_flux_source_phase_0")

        else:  # self.ell == 1:
            source = source_phase_1
            source.set_name("interface_fluid_mass_flux_source_phase_1")

        eq = self.balance_equation(
            subdomains, accumulation, flux, source, dim=1
        )  # * pp.ad.Scalar(1e6)
        eq.set_name("mass_balance_equation")

        return (
            eq,
            accumulation,
            None,
            None,
            flux_no_intf,
            flux_intf_phase_0,
            flux_intf_phase_1,
            source_phase_0,
            source_phase_1,
        )

    def mass_balance_equation(self, subdomains: list[pp.Grid]) -> pp.ad.Operator:
        eq, _, _, _, _, _, _, _, _ = self.eq_fcn_mass(subdomains)

        return eq


    def ppu_discretization(
        self, subdomains: list[pp.Grid], flux_array_key
    ) -> pp.ad.UpwindAd:
        """
        flux_array_key =  either darcy_flux_phase_0 or darcy_flux_phase_1
        """
        return pp.ad.UpwindAd(self.ppu_keyword + "_" + flux_array_key, subdomains, flux_array_key)

class PartialFinalModel(
    test_hu_model.PrimaryVariables,
    EquationsPPU,
    test_hu_model. ConstitutiveLawPressureMass,
    test_hu_model.BoundaryConditionsPressureMass,
    test_hu_model.SolutionStrategyPressureMass,
    test_hu_model.MyModelGeometry,
    pp.DataSavingMixin,
):
    """ """


if __name__ == "__main__":

    class FinalModel(PartialFinalModel):  # I'm sorry...
        def __init__(self, mixture, params: Optional[dict] = None):
            super().__init__(params)
            self.mixture = mixture
            self.ell = 0 # 0 = wetting, 1 = non-wetting
            self.gravity_value = 1  # pp.GRAVITY_ACCELERATION
            self.dynamic_viscosity = 1  # TODO: it is hardoced everywhere, you know...

            self.xmax = 1 * 1
            self.ymax = 1 * 1 

            self.relative_permeability = pp.tobedefined.relative_permeability.rel_perm_quadratic
            self.mobility = pp.tobedefined.mobility.mobility(self.relative_permeability) 
            self.mobility_operator = pp.tobedefined.mobility.mobility_operator(self.mobility)

    wetting_phase = pp.composite.phase.Phase(rho0=1)
    non_wetting_phase = pp.composite.phase.Phase(rho0=0.5)

    mixture = pp.Mixture()
    mixture.add([wetting_phase, non_wetting_phase])

    params = test_hu_model.params
    model = FinalModel(mixture, params) # why... add it directly as attribute: model.mixture = mixture

    pp.run_time_dependent_model(model, params)
