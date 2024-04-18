#!/usr/bin/env python
# coding: utf-8

# # Introduction
# This introduction describes PorePy's main concepts and features. For more detail on individual topics and code examples, we refer the reader to the corresponding tutorial notebooks, see the [readme](https://github.com/pmgbergen/porepy/blob/develop/Readme.md) for an overview.
# For a more rigorous and theoretical description, please see [Keilegavlen et al.](https://link.springer.com/article/10.1007/s10596-020-10002-5) and references therein.
# 
# ## Conceptual model
# PorePy is a simulation toolbox covering several (coupled) physical processes tailored to _fractured_ porous media. 
# The underlying mixed-dimensional model arises from dimension reduction of fractures, which is motivated by their high aspect ratio. 
# Averaging along the thickness of the fracture allows representing it as a co-dimension one subdomain where the aperture [m] accounts for the collapsed dimension.
# The dimension reduction leads to a mixed-dimensional model of the rock, where the host rock, fractures, fracture intersections and intersections of fracture intersections are treated as individual subdomains.
# 
# In practice this means that for a three-dimensional domain, the fractures within it are two-dimensional planes.
# Similarly, fracture intersections are represented by one-dimensional lines and intersections of fracture intersections by zero-dimensional points. 
# The same principle holds also for two-dimensional domains: the fractures are one-dimensional lines and fracture intersections are zero-dimensional points. 
# To account for the collapsed area and volume, we introduce a _specific volume_ [m$^{3-d}$], with $d$ denoting the dimension of the subdomain in question. 
# For more information regarding the specific volume we refer to the [conventions](https://github.com/pmgbergen/porepy/blob/develop/tutorials/conventions.ipynb) tutorial.
# 
# Finally, in the mixed-dimensional model, each lower-dimensional subdomain corresponds to an internal boundary for one or more higher-dimensional subdomains. 
# Each such pair of neighboring subdomains exactly one dimension apart is connected through an _interface_. This is illustrated in the figure below, for the 2D subdomain $\Omega_h$, the 1D subdomain $\Omega_l$ and the 1D interface $\Gamma_j$ (figure taken from [Keilegavlen et al.](https://link.springer.com/article/10.1007/s10596-020-10002-5)).
# 
# <img src='img/subdomains_interfaces.png' width=600>
# 
# Thus, the entire mixed-dimensional domain can be represented by a graph with *nodes* corresponding to subdomains and *edges* corresponding to interfaces.
# In the code these components are normally referred to as subdomains and interfaces.
# 
# ## Key components
# The definition of a mixed-dimensional domain allows modeling, equation definition and numerical discretization to take place on individual subdomains and interfaces.
# From the numerical point of view, this facilitates a high level of modularity and extensive code reuse. 
# It also implies the need for a structure representing a mixed-dimensional grid and handling of this structure:
# * The `MixedDimensionalGrid` object has a graph structure whose nodes are $d$-dimensional grids representing the subdomains and whose edges correspond to interface (or "mortar") grids. Each node and edge also has an associated data dictionary for storage of parameters, variables, discretization matrices etc.
# * A `FractureNetwork` class processes a set of fractures (defined through vertices) by computing intersections to define the collection of subdomains. Then, the actual meshing of subdomains is done using [gmsh](https://gmsh.info/). The resulting mixed-dimensional mesh is conforming, in that any internal boundary corresponds to a face of the mesh, but may be non-matching, i.e. we do not require a one-to-one face correspondance between the two sides of a fracture.
# * The `EquationSystem` class handles variables and equations. This includes bookkeeping of user-defined equations, as well as assembling the linear system of equations during a Newton iteration. 
# * An automatic differentiation (AD) package handles the assembly of the Jacobian matrix during a Newton iteration. Due to this, equations are defined as AD operators - that is, operators on which automatic differentiation can be performed. This is achieved by defining all of the building blocks of the equation to be AD operators, including the independent variables. All of this is handled by the `Operator` class.
# * `Discretization` objects are defined on each subdomain and interface. Discretization is local to a subdomain and relies on the corresponding grid and parameters defined in the subdomain's data dictionary. To facilitate coupling over the interfaces, subdomain discretizations operate on grids and data of the two neighboring subdomains as well as those of the interface. 
# 
# ## Capabilities
# The above outlined framework enables implementation of numerical models for a wide range of phenomena in fractured porous media. Specifically, the code has been used to simulate:
# * Single-phase flow, both incompressible and compressible.
# * Two-phase flow
# * Linear elasticity
# * Fracture contact mechanics
# * Flow and transport, either decoupled, weakly coupled or strongly coupled.
# * Biot poroelasticity
# * Thermo-poroelasticity
# 
# Combinations of the above are also possible, e.g. solving the Biot equations in the matrix, and flow and contact mechanics in fractures.
# The maturity of the code varies between the components, and adaptations may be needed to apply PorePy to a specific problem.
# 
# The setup of a coupled multi-physics mixed-dimensional simulation is nontrivial. 
# We therefore provide `Model` classes with predefined default parameters, discretizations, equation definitions etc. A simulation can then be set up quite easily by modularly adjusting the model class according to the user's needs.
# The models make use of multiple inheritance, and the user is therefore recommended to familiarize themselves with this concept in Python.
# Several possible model setups are demonstrated in other tutorials.
# 
# 
# ## Other comments
# If you are considering to apply PorePy to a specific problem, you may want to take the following comments into consideration:
# * The software can only be applied by writing Python code. The amount and complexity of coding needed depends on the problem at hand.
# * Problems are defined in terms of governing equations, often in the form of partial differential equations. New models, for instance constitutive relations, to be implemented in PorePy must adhere to this framework.
# * There is currently no native support for linear and non-linear solvers in PorePy; specifically tailored approaches for coupled physical processes are lacking. This means simulations on large grids will be time consuming.
