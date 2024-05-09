import sys
import os
import pdb

import numpy as np

sys.path.remove("/home/inspiron/Desktop/PhD/porepy/src")
sys.path.append("/home/inspiron/Desktop/PhD/eni_venv/porepy/src")
import porepy as pp
from porepy.applications.md_grids.domains import nd_cube_domain
from porepy.models.fluid_mass_balance import SinglePhaseFlow
from porepy.models.momentum_balance import MomentumBalance


print("pp.__file__ = ", pp.__file__)


class ModifiedGeometry:
    def set_domain(self) -> None:
        """Defining a two-dimensional square domain with sidelength 2."""
        size = self.solid.convert_units(2, "m")
        self._domain = nd_cube_domain(2, size)

    def grid_type(self) -> str:
        """Choosing the grid type for our domain."""
        return self.params.get("grid_type", "cartesian")

    def meshing_arguments(self) -> dict:
        """Meshing arguments for md-grid creation."""
        cell_size = self.solid.convert_units(0.25, "m")
        mesh_args: dict[str, float] = {"cell_size": cell_size}
        return mesh_args


# for x in dir(pp.models.fluid_mass_balance.BoundaryConditionsSinglePhaseFlow):
#     if x.startswith("bc_values") or x.startswith("bc_type"):
#         print(x)


# class SinglePhaseFlowExample1(ModifiedGeometry, SinglePhaseFlow):
#     def bc_type_fluid_flux(self, sd: pp.Grid) -> pp.BoundaryCondition:
#         """Setting the Dirichlet type on the east boundary, Neumann elsewhere."""
#         domain_sides = self.domain_boundary_sides(sd)
#         return pp.BoundaryCondition(sd, faces=domain_sides.east, cond="dir")

#     def bc_values_fluid_flux(self, boundary_grid: pp.BoundaryGrid) -> np.ndarray:
#         """Setting the values of the fluid mass flux."""
#         mass_flux_vals = np.zeros(boundary_grid.num_cells)

#         domain_sides = self.domain_boundary_sides(boundary_grid)
#         influx_cells = np.zeros(boundary_grid.num_cells, dtype=bool)
#         influx_cells[domain_sides.west] = True
#         # Setting the values on the west boundary where 0.5 < y < 1.5
#         # "&" operator is the elementwise boolean AND
#         influx_cells &= boundary_grid.cell_centers[1] > 0.5
#         influx_cells &= boundary_grid.cell_centers[1] < 1.5

#         mass_flux_vals[influx_cells] = self.fluid.convert_units(-1, "kg*s^-1")
#         return mass_flux_vals

#     def bc_values_pressure(self, boundary_grid: pp.BoundaryGrid) -> np.ndarray:
#         pressure_vals = np.zeros(boundary_grid.num_cells)
#         domain_sides = self.domain_boundary_sides(boundary_grid)
#         pressure_vals[domain_sides.east] = self.fluid.convert_units(5, "Pa")
#         return pressure_vals


# single_phase_flow = SinglePhaseFlowExample1(params={})
# pp.run_time_dependent_model(single_phase_flow, params={})
# pp.plot_grid(single_phase_flow.mdg, single_phase_flow.pressure_variable, plot_2d=True)


# class SinglePhaseFlowExample2(ModifiedGeometry, SinglePhaseFlow):
#     # Note that now this is bc_type_darcy_flux, not the bc_type_fluid_flux.
#     def bc_type_darcy_flux(self, sd: pp.Grid) -> pp.BoundaryCondition:
#         """Everything is the same as in the previous example."""
#         domain_sides = self.domain_boundary_sides(sd)
#         return pp.BoundaryCondition(sd, faces=domain_sides.east, cond="dir")

#     # Note that now this is bc_values_darcy_flux, not the bc_values_fluid_flux.
#     def bc_values_darcy_flux(self, boundary_grid: pp.BoundaryGrid) -> np.ndarray:
#         """Setting the Darcy flux values on the west boundary."""
#         darcy_flux_vals = np.zeros(boundary_grid.num_cells)

#         # Same as in the previous example
#         domain_sides = self.domain_boundary_sides(boundary_grid)
#         influx_cells = np.zeros(boundary_grid.num_cells, dtype=bool)
#         influx_cells[domain_sides.west] = True
#         influx_cells &= boundary_grid.cell_centers[1] > 0.5
#         influx_cells &= boundary_grid.cell_centers[1] < 1.5

#         # The value is the same
#         darcy_flux_vals[influx_cells] = self.fluid.convert_units(-1, "Pa*m^-1")
#         return darcy_flux_vals

#     # This method did not change.
#     def bc_values_pressure(self, boundary_grid: pp.BoundaryGrid) -> np.ndarray:
#         """Everything is the same as in the previous example."""
#         pressure_vals = np.zeros(boundary_grid.num_cells)
#         domain_sides = self.domain_boundary_sides(boundary_grid)
#         pressure_vals[domain_sides.east] = self.fluid.convert_units(5, "Pa")
#         return pressure_vals


# # We modify the fluid viscosity.
# fluid_constants = pp.FluidConstants(
#     {
#         "viscosity": 10,  # 10 times larger than in the previous example.
#     }
# )
# single_phase_flow = SinglePhaseFlowExample2(
#     params={"material_constants": {"fluid": fluid_constants}}
# )
# pp.run_time_dependent_model(single_phase_flow, params={})
# pp.plot_grid(single_phase_flow.mdg, single_phase_flow.pressure_variable, plot_2d=True)


class ModifiedBoundaryConditions:
    def bc_type_mechanics(self, sd: pp.Grid) -> pp.BoundaryConditionVectorial:
        """Set boundary condition type for the problem."""
        bounds = self.domain_boundary_sides(sd)

        # Set the type of west and east boundaries to Dirichlet. North and south are
        # Neumann by default.
        bc = pp.BoundaryConditionVectorial(sd, bounds.west + bounds.east, "dir")
        return bc

    def bc_values_stress(self, bg: pp.BoundaryGrid) -> np.ndarray:
        """Setting stress boundary condition values at north and south boundaries.

        Specifically, we assign different values for the x- and y-component of the
        boundary value vector.
        """
        values = np.ones((self.nd, bg.num_cells))
        bounds = self.domain_boundary_sides(bg)

        # Assigning x-component values
        values[0][bounds.north + bounds.south] *= self.solid.convert_units(4.5, "Pa")

        # Assigning y-component values
        values[1][bounds.north + bounds.south] *= self.solid.convert_units(0.5, "Pa")

        return values.ravel("F")

    def bc_values_displacement(self, bg: pp.BoundaryGrid) -> np.ndarray:
        """Setting displacement boundary condition values.

        This method returns an array of boundary condition values with the value 5t for
        western boundaries and ones for the eastern boundary.

        """
        t = self.time_manager.time

        values = np.zeros((self.nd, bg.num_cells))
        bounds = self.domain_boundary_sides(bg)

        values[0][bounds.west] += self.solid.convert_units(5.0 * t, "m")
        values[0][bounds.east] += self.solid.convert_units(1.0, "m")

        return values.ravel("F")


class MomentumBalanceTimeDependentBC(
    ModifiedGeometry, ModifiedBoundaryConditions, MomentumBalance
): ...


for x in dir(MomentumBalanceTimeDependentBC):
    if x.startswith("bc_values") or x.startswith("bc_type"):
        print(x)


# Set final time, amount of time-steps and the time-step size
final_time = 10.0
time_steps = 10.0
dt = final_time / time_steps

# Instantiate pp.TimeManager with the information provided above
time_manager = pp.TimeManager(
    schedule=[0.0, final_time],
    dt_init=dt,
    constant_dt=True,
)

# Include the time_manager to the params dictionary
params = {"time_manager": time_manager}

model = MomentumBalanceTimeDependentBC(params)
pp.run_time_dependent_model(model=model, params=params)


print("\nDone!")
