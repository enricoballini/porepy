#!/usr/bin/env python
# coding: utf-8

# # Conventions

# This tutorial lists some of the most important conventions and assumptions used in PorePy.
# 
# Specifically, we cover conventions for:
# * Geometry
# * Boundaries
# * Equations
# * Apertures
# * Coupling between dimensions

# ## Geometry
# Normal vectors are defined once for each face and can be accessed through `sd.face_normals` for a subdomain, `sd`. These vectors are weighted with the face area, meaning that the area of a face can be found by taking the norm of its normal vector.
# 
# For a subdomain grid sd, the field `sd.cell_faces` is the discrete divergence: 
# * It has dimensions (number of faces) $\times$ (number of cells)
# * It is nonzero only for the neighboring faces of any given cell
# * The value is positive if the normal vector of the face points outwards from the cell and negative otherwise.
# 
# See [this](./grid_topology.ipynb) tutorial for a demonstrations of the aforementioned.
# 
# 
# This can be used to obtain the outwards pointing normal vector of a cell. We simply multiply with the value ($\pm 1$) of sd.cell_faces. For instance, for cell `c` and face `f` we have
# 
# `outward_normal = sd.cell_faces[f, c] * sd.face_normals[:, f]`

# ## Boundaries
# ### Flow
# Outflow values are considered positive for Neumann type boundary conditions, i.e. outward pointing normal vectors are assumed.
# 
# On inner boundaries, a positive mortar flux $\lambda$ implies flow from the higher-dimensional to the lower-dimensional domain.
# ### Mechanics
# Mechanical boundary condtions are always given in global coordinates. 
# For Dirichlet boundary condition values, we prescribe the displacement values directly without regarding the normal vector. 
# For a 2d  domain, this means that a prescribed boundary value of $\mathbf{u}_b = [1, -2]$ signifies a unitary displacement to the right, and downwards with magnitude 2. 
# Similarly, $\sigma\cdot \mathbf{n} = [-1, 0]$ implies a unitary force to the left.
# 
# Displacements defined on inner boundaries are also in global coordinates.

# ## Equations
# ### Momentum balance
# The momentum balance equation within PorePy follows the convention of positive tensile stress. 
# In practice this means that the equation is described by the following formulation:
# 
# $\frac{\partial^2 u}{\partial t^2} = \nabla \cdot \sigma (\epsilon (u)) + q$,
# 
# where $u$ is displacement, $\sigma(\epsilon(u))$ is the stress tensor and $q$ is the body force.
# Usually the stress tensor is related to displacement through Hooke's law and the strain-displacement relation for small deformations:
# 
# $\sigma(\epsilon(u)) = C : \epsilon(u) = C : \frac{1}{2} (\nabla u + (\nabla u)^T)$,
# 
# where $C$ denotes the stiffness tensor.

# ## Apertures
# In our mixed-dimensional models we denote the thickness of the fractures by the term `aperture` [m], denoted by $a$. 
# 
# Volumes [m$^3$] of fracture or intersection cells are computed through their specific volumes. 
# In any given dimension d, this is the volume of the cell per d-dimensional measure. 
# In 2d, it is the volume per area, and has dimensions [m$^3/$m$^2$] = [m]. 
# The measure is a line measure for 1d intersection lines, and the specific volume thus has the dimensions [m$^3/$m] = [m$^2$]. 
# In general, the dimensions are [m$^{3-d}$], where d is the dimension of the subdomain in question.
# The full volume is always the product of the d-dimensional cell volume and the specific volume.
# 
# `volumes = g.cell_volumes * specific_volumes` $\quad$ [m$^3$]
# 
# This implies that some parameters should be weighted by the specific volumes. 
# This holds for tangential permeability, volumetric source terms and several other parameters. 
# To be clear, the permeabilities specified within and between subdomains are absolute, that is, no aperture scaling of permeabilities takes place inside the code.

# ## Coupling between dimensions
# ### Diffusion
# A diffusive flux between dimensions is typically discretized using the `RobinCoupling` (see  numerics/interface_laws/elliptic_interface_laws.py). Its `discretize` function requires the parameter "normal_diffusivity" ($k_n$ in the context of flow, where it is related to the normal permeability of the fracture) to be defined for each mortar cell, to be used according to the Darcy type interface law
# $\lambda = - k_n (\check p - \hat p $).
# 
# Example: If one wishes to impose a normal permeability of $1/10$, that is a fracture which has a somewhat blocking effect on the flow if the matrix permeability is unitary, one should scale it by the aperture $a$ (see the Apertures section):
# 
# $k_n = 1/10 \cdot \frac{1}{(a/2)} = 1/10 \cdot \frac{2}{a}$.
# 
# The factor $a/2$ is the collapsed distance between the mortar and the center of the fracture, and may be thought of as the distance component of the discrete normal gradient at the fracture, i.e.:
# 
# $ \nabla_{normal, discrete} \,p = \frac{\check p - \hat p }{a/2} $
# 
