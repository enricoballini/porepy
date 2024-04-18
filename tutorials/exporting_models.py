#!/usr/bin/env python
# coding: utf-8

# # Exporting in models
# 
# This tutorial provides inspiration and potentially useful code for exporting data in a `pp.Model` based simulation for visualization in e.g. [ParaView](https://www.paraview.org/). While the PorePy [model class](./single_phase_flow.ipynb) and the [exporter](./exporter.ipynb) are introduced elsewhere, this tutorial provides some detail on how to combine them. To this end, we will choose the transient model for mixed-dimensional poroelasticity with fracture deformation and adjust it to make sure the solution is exported according to our requirements, i.e. at the right stages of the simulation and that we export the right fields/variables. The tutorial consists of two parts. The first covers standard usage and the second covers more advanced usage related to debugging. 
# 
# We start with a very simple case, indicating which methods could be overwritten to specify which variables are exported. The starting point is the `DataSavingMixin` class, which is responsible for all things related to saving and exporting of data during a simulation. By default, it exports all primary variables as well as apertures and specific volumes, see `data_to_export`. Exporting is performed at the start of the simulation and at the end of each time step.

# In[12]:


import porepy as pp
import numpy as np
from porepy.models.poromechanics import Poromechanics


params = {"folder_name": "model_exporting"}

# The compound class Poromechanics inherits from DataSavingMixin.
model_0 = Poromechanics(params)
pp.run_time_dependent_model(model_0, params)


# Since the default `set_geometry` method of `Poromechanics` (inherited from `ModelGeometry`) produces a monodimensional domain, we get data files containing pressure and displacement in the matrix domain and no tractions (no fractures are present). 
# 
# 
# ## Mixed-dimensional simulations
# We now extend to the mixed-dimensional case. In the parameters, we adjust the file name to avoid overwriting the previous files. If you inspect the suffixes of the files created, you can see how the exporter deals with multiple time steps by default.

# In[13]:


from porepy.applications.md_grids.model_geometries import (
    SquareDomainOrthogonalFractures,
)


class MDPoroelasticity(
    SquareDomainOrthogonalFractures,
    Poromechanics,
):
    """Combine the geometry class with the poromechanics class."""

    pass


params.update(
    {
        "end_time": 2,
        "file_name": "md",
    }
)
model_1 = MDPoroelasticity(params)
pp.run_time_dependent_model(model_1, params)


# ### Tailored exporting
# Suppose we want to perform a simulation similar to above, but require more data for visualization. 
# For instance, we might very reasonably want to look at the displacement jump on the fractures. 
# This is not a primary variable, and thus not exported by default. 
# We implement this as a second mixin class which we combine with `MDPoroelasticity`which adds a tuple containing grid, name and values to the list of data to be exported. 
# For other allowed formats, see [exporter](./exporter.ipynb).

# In[14]:


class DisplacementJumpExporting:
    def data_to_export(self):
        """Define the data to export to vtu.

        Returns:
            list: List of tuples containing the subdomain, variable name,
            and values to export.

        """
        data = super().data_to_export()
        for sd in self.mdg.subdomains(dim=self.nd - 1):
            vals = self._evaluate_and_scale(sd, "displacement_jump", "m")
            data.append((sd, "displacement_jump", vals))
        return data


class TailoredPoroelasticity(DisplacementJumpExporting, MDPoroelasticity):
    """Combine the exporting class with the poromechanics class."""

    pass


params.update(
    {
        "end_time": 2,
        "file_name": "jumps",
    }
)
model_2 = TailoredPoroelasticity(params)
pp.run_time_dependent_model(model_2, params)


# # Iteration exporting for debugging
# We now turn to exporting data for each iteration when solving the nonlinear system. This second part is significantly more advanced than the preceeding part and some users may want to skip it.
# 
# Exporting iterations can be quite handy when debugging or trying to make sense of why your model doesn't converge. Moreover, even when everything works as a dream, you might want to visualize how convergence is reached, for instance to distinguish between global and local effects. 
# 
# We stress that not only which variables to export but also when you wish to export them may vary between applications. In the model provided below, we export at all iterations using a separate exporter, keeping track of time step and iteration number using the vtu file suffix and collecting them using a single pvd file. The "time step" suffix of an iteration file is the sum of the iteration index and the product of the current time step index and $r$. Here $r$ is the smallest power of ten exceeding the maximum number of non-linear iterations. 
# 
# Expecting that the simulation may crash or be stopped at any point, we (over)write a pvd file each time a new vtu file is added. Alternative approaches and refinements include writing one pvd file for each time step and writing debugging files on some condition, e.g. that the iteration index exceeds some threshold.

# In[15]:


class IterationExporting:
    def initialize_data_saving(self):
        """Initialize iteration exporter."""
        super().initialize_data_saving()
        # Setting export_constants_separately to False facilitates operations such as
        # filtering by dimension in ParaView and is done here for illustrative purposes.
        self.iteration_exporter = pp.Exporter(
            self.mdg,
            file_name=self.params["file_name"] + "_iterations",
            folder_name=self.params["folder_name"],
            export_constants_separately=False,
        )

    def data_to_export_iteration(self):
        """Returns data for iteration exporting.

        Returns:
            Any type compatible with data argument of pp.Exporter().write_vtu().

        """
        # The following is a slightly modified copy of the method
        # data_to_export() from DataSavingMixin.
        data = []
        variables = self.equation_system.variables
        for var in variables:
            # Note that we use iterate_index=0 to get the current solution, whereas
            # the regular exporter uses time_step_index=0.
            scaled_values = self.equation_system.get_variable_values(
                variables=[var], iterate_index=0
            )
            units = var.tags["si_units"]
            values = self.fluid.convert_units(scaled_values, units, to_si=True)
            data.append((var.domain, var.name, values))
        return data

    def save_data_iteration(self):
        """Export current solution to vtu files.

        This method is typically called by after_nonlinear_iteration.

        Having a separate exporter for iterations avoids distinguishing between iterations
        and time steps in the regular exporter's history (used for export_pvd).

        """
        # To make sure the nonlinear iteration index does not interfere with the
        # time part, we multiply the latter by the next power of ten above the
        # maximum number of nonlinear iterations. Default value set to 10 in
        # accordance with the default value used in NewtonSolver
        n = self.params.get("max_iterations", 10)
        p = round(np.log10(n))
        r = 10**p
        if r <= n:
            r = 10 ** (p + 1)
        self.iteration_exporter.write_vtu(
            self.data_to_export_iteration(),
            time_dependent=True,
            time_step=self._nonlinear_iteration + r * self.time_manager.time_index,
        )

    def after_nonlinear_iteration(self, solution_vector: np.ndarray) -> None:
        """Integrate iteration export into simulation workflow.

        Order of operations is important, super call distributes the solution to
        iterate subdictionary.

        """
        super().after_nonlinear_iteration(solution_vector)
        self.save_data_iteration()
        self.iteration_exporter.write_pvd()


class IterationCombined(IterationExporting, TailoredPoroelasticity):
    """Add iteration exporting to the tailored poroelasticity class."""

    pass


params.update(
    {
        "end_time": 2,
        "file_name": "iterations",
    }
)
model_3 = IterationCombined(params)
pp.run_time_dependent_model(model_3, params)

