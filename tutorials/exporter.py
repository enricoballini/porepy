#!/usr/bin/env python
# coding: utf-8

# # Introduction
# Currently, the standard procedure within PorePy is to export data to vtu and pvd format for visualization with ParaView. 
# This tutorial explains how to use the `Exporter`. 
# In particular, it will showcase different ways to address data, how constant-in-time data is handled, and how pvd-files are managed. 
# 
# First, an example data set is defined, then the actual exporter is defined, before all supported ways to export data are demonstrated.
# 
# <b>Note:</b> Related but not necessary for this tutorial: it is highly recommended to read the ParaView documentation. 

# ## Example contact mechanics model for a mixed-dimensional geometry
# In order to illustrate the capability and explain the use of the Exporter, we consider a ContactMechanicsBiot model for a two-dimensional fractured geometry. 
# The mixed-dimensional geometry consists of a 2D square and two crossing 1D fractures.

# In[16]:


import numpy as np
import porepy as pp
from porepy.models.derived_models.biot import BiotPoromechanics
from porepy.applications.md_grids.model_geometries import (
    SquareDomainOrthogonalFractures,
)


class BiotFractured(SquareDomainOrthogonalFractures, BiotPoromechanics):
    def meshing_arguments(self) -> dict:
        cell_size = self.solid.convert_units(0.25, "m")
        return {"cell_size": cell_size}


params = {"fracture_indices": [0, 1]}
model = BiotFractured(params)
model.prepare_simulation()


# The default data of the model is stored as `pp.TIME_STEP_SOLUTIONS` in the mixed-dimensional grid.
# Let's have a look:

# In[17]:


# Determine all keys of all states on all subdomains
subdomain_states = []
for sd, data in model.mdg.subdomains(return_data=True):
    subdomain_states += data[pp.TIME_STEP_SOLUTIONS].keys()
print("Keys of the states defined on subdomains:", set(subdomain_states))

# Determine all keys of all states on all interfaces
interface_states = []
for sd, data in model.mdg.interfaces(return_data=True):
    interface_states += data[pp.TIME_STEP_SOLUTIONS].keys()
print("Keys of the states defined on interfaces:", set(interface_states))


# ## Defining the exporter
# Two arguments are required to define an object of type pp.Exporter: a mixed-dimensional grid, and the target name of the output. 
# Optionally, one can add a directory name, and instead of a mixed-dimensional grid, single grids can also be provided (see Example 7).

# In[18]:


exporter = pp.Exporter(model.mdg, file_name="file", folder_name="folder")


# In the following, we showcase how to use the main subroutines for exporting data:
# - write_vtu()
# - write_pvd()
# 
# The former addresses the export of data for a specific time step, while the latter gathers the previous exports and collects them in a single file. 
# This allows an easier analysis in ParaView.

# ## Example 1: Exporting states
# Data stored in the mixed-dimensional grid under `pp.TIME_STEP_SOLUTIONS` can be simply exported by addressing their keys using the routine `write_vtu()`. 
# We define a dedicated exporter for this task.

# In[19]:


exporter_1 = pp.Exporter(model.mdg, file_name="example-1",folder_name="exporter-tutorial") 
exporter_1.write_vtu([
    model.pressure_variable,
    model.displacement_variable,
    model.interface_darcy_flux_variable
    ])


# <b>Note:</b> Here all available representations (i.e., on all dimensions) of the states will be exported.

# ## Example 2: Exporting states on specified grids
# Similar to Example 1, we will again export states by addressing their keys, but target only a subset of grids. 
# For instance, we fetch the grids for the subdomains and interface.
# 
# <b>Note:</b> For now, one has to make sure that subsets of the mixed-dimensional grid contain all grids of a particular dimension.

# In[20]:


subdomains_1d = model.mdg.subdomains(dim=1)
subdomains_2d = model.mdg.subdomains(dim=2)
interfaces_1d = model.mdg.interfaces(dim=1)


# And as a simple example extract the 2D subdomain:

# In[21]:


sd_2d = subdomains_2d[0]


# We export pressure on all 1D subdomains, displacements on all 2D subdomains, and the mortar pressures on all interfaces. 
# For this, we use tuples of grid(s) and keys. 
# In order to not overwrite the previous data, we define a new exporter.

# In[22]:


exporter_2 = pp.Exporter(model.mdg, "example-2", "exporter-tutorial")
exporter_2.write_vtu([
    (subdomains_1d, model.pressure_variable), 
    (subdomains_2d, model.displacement_variable),
    (interfaces_1d, model.interface_darcy_flux_variable),
    ])


# ## Example 3: Exporting explicitly defined data
# We can also export data which is not stored in the mixed-dimensional grid under `pp.TIME_STEP_SOLUTIONS`. 
# This capability requires defining tuples of (1) a single grid, (2) a key, and (3) the data vector. 
# For example, let's export the cell centers of the 2D subdomain 'sd_2d', as well as all interfaces (with different signs for the sake of the example). 
# Again, we define a dedicated exporter for this task.
# 

# In[23]:


subdomain_data = [(sd_2d, "cc", sd_2d.cell_centers)]
interface_data = [
    (intf, "cc_e", (-1) ** i * intf.cell_centers)
    for i, intf in enumerate(interfaces_1d)
]
exporter_3 = pp.Exporter(model.mdg, "example-3", "exporter-tutorial")
exporter_3.write_vtu(subdomain_data + interface_data)


# ## Example 4: Flexibility in the input arguments
# The export allows for an arbitrary combination of all previous ways to export data.
# 
# Specifically, this example shows how to do the following export combination:
# * The "custom" data vector on the `sd_2d` from example 3
# * The pressure variable only for `subdomain_1d`
# * All available representations of the displacement
# * The "custom" interface data from example 3

# In[24]:


exporter_4 = pp.Exporter(model.mdg, "example-4", "exporter-tutorial")
exporter_4.write_vtu(
    [
        (sd_2d, "cc", sd_2d.cell_centers),
        (subdomains_1d, model.pressure_variable), 
        model.displacement_variable] + interface_data
    )


# ## Example 5: Exporting data in a time series
# Data can also be exported in a time series, and the Exporter takes care of managing the file names. 
# The user will only have to prescribe the time step number, and here we consider a time series consisting of 5 steps. 
# For simplicity, we look at an analogous situation as in Example 1.

# In[25]:


exporter_5 = pp.Exporter(model.mdg, "example-5", "exporter-tutorial")
variable_names = [
    model.pressure_variable,
    model.displacement_variable,
    model.interface_darcy_flux_variable,
    ]
for step in range(5):
    # Data may change
    exporter_5.write_vtu(variable_names, time_step=step)


# Alternatively, one can also let the Exporter internally manage the stepping and the appendix used when storing the data to file. 
# This is triggered by the keyword `time_dependent`.

# In[26]:


exporter_5 = pp.Exporter(model.mdg, "example-5", "exporter-tutorial")
for step in range(5):
    # Data may change
    exporter_5.write_vtu(variable_names, time_dependent=True)


# ## Example 6: Exporting constant data
# The export of both grid and geometry related data as well as heterogeneous material parameters may be of interest. 
# However, these often change very seldomly in time or are even constant in time. 
# In order to save storage space, constant data is stored separately. 
# A multirate approach is used to address slowly changing "constant" data, which results in an extra set of output files. 
# Every time constant data has to be updated (in a time series), the output files are updated as well. 

# In[27]:


exporter_6_a = pp.Exporter(model.mdg, "example-6-a", "exporter-tutorial")
# Define some "constant" data
exporter_6_a.add_constant_data([(sd_2d, "cc", sd_2d.cell_centers)])
for step in range(5):
    # Update the previously defined "constant" data
    if step == 2:
        exporter_6_a.add_constant_data([(sd_2d, "cc", -sd_2d.cell_centers)])
    # All constant data will be exported also if not specified
    exporter_6_a.write_vtu(variable_names, time_step=step)


# The default is that constant data is always printed to extra files. 
# Since the vtu format requires geometrical and topological information on the mesh (points, connectivity etc.), this type of constant data is exported to each vtu file. 
# Depending on the situation, this overhead can be significant. 
# Thus, one can also choose to print the constant data to the same files as the standard data, by setting a keyword when defining the exporter. 
# With a similar setup as in part A (just above), the same output is generated, but managed differently among files.

# In[28]:


exporter_6_b = pp.Exporter(model.mdg, "example-6-b", "exporter-tutorial", export_constants_separately = False)
exporter_6_b.add_constant_data([(sd_2d, "cc", sd_2d.cell_centers)])
for step in range(5):
    # Update the previously defined "constant" data
    if step == 2:
        exporter_6_b.add_constant_data([(sd_2d, "cc", -sd_2d.cell_centers)])
    # All constant data will be exported also if not specified
    exporter_6_b.write_vtu(variable_names, time_step=step)


# ## Example 5 revisisted: PVD format
# The pvd format collects previously exported data. 
# At every application of `write_vtu` a corresponding pvd file is generated. 
# This file gathers all "vtu" files correpsonding to this time step. It is recommended to use the "pvd" file for analyzing the data in ParaView.
# 
# In addition, when considering a time series, it is possible to gather data connected to multiple time steps and assign the actual time to each time step. 
# Assume that Example 5 corresponds to an adaptive time stepping. 
# We define the actual times, and collect the exported data from Example 5 in a single pvd file.

# In[29]:


times_5 = [0.0, 0.1, 1.0, 2.0, 10.0]
exporter_5.write_pvd(times_5)


# When providing no argument to `write_pvd()`, the time steps are used as actual times.

# ## Example 7: Exporting data on a single grid
# It is also possible to export data without prescribing a mixed-dimensional grid, but a single grid. In this case, one has to assign the data when writing to vtu. For this, a key and a data array (with suitable size) have to be provided.

# In[30]:


exporter_7 = pp.Exporter(sd_2d, "example-7", "exporter-tutorial")
exporter_7.write_vtu([("cc", sd_2d.cell_centers)])


# # What we have explored
# Both mixed-dimensional and single grids in PorePy, in addition to their related information such as solutions and parameters, can be exported to the vtu format.
# Further they can be visualized in ParaView (or other compatible software). 
# The key object is `pp.Exporter` and its `write_vtu` method, which allow for several modes of exporting information.
