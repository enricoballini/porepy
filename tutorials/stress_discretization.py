#!/usr/bin/env python
# coding: utf-8

# # Multi-point stress approximation (MPSA)

# PorePy includes a multi-point stress approximation discretization for the linear elasticity problem:
# \begin{equation}
# \nabla\cdot \sigma = -\vec f,\quad \vec x \in \Omega
# \end{equation}
# where $\vec f$ is a body force, and the stress $\sigma$ is given as a linear function of the displacement
# \begin{equation}
# \sigma = C: \nabla \vec u.
# \end{equation}
# 
# The convention in PorePy is that tension is positive. This means that the Cartesian component of the traction $\vec T = \sigma \cdot \vec n$, for a direction $\vec r$ is positive number if the inner product $\vec T\cdot \vec r$ is positive. The displacements will give the difference between the initial state of the rock and the deformed state. If we consider a point in its initial state $\vec x \in \Omega$ and let $\vec x^* \in \Omega$ be the same point in the deformed state, to be consistent with the convention we used for traction, the displacements are given by $\vec u = \vec x^* - \vec x$, that is, $u$ points from the initial state to the finial state.
# 
# To close the system we also need to define a set of boundary conditions. Here we have three posibilities, Neumman conditions, Dirichlet conditions or Robin conditions, and we divide the boundary into three disjont sets $\Gamma_N$, $\Gamma_D$ and $\Gamma_R$ for the three different types of boundary conditions
# \begin{equation}
# \vec u = g_D, \quad \vec x \in \Gamma_D\\
# \sigma\cdot n = g_N, \quad \vec x \in \Gamma_N\\
# \sigma\cdot n + W \vec u= g_R,\quad \vec x\in \Gamma_R.
# \end{equation}

# To solve this system we first have to create the grid.

# In[1]:


import numpy as np
import porepy as pp

# Create grid
n = 5
g = pp.CartGrid([n, n])
g.compute_geometry()


# We also need to define the stress tensor $C$. In PorePy, the constitutive law
# \begin{equation}
# \sigma = C:\nabla u = 2  \mu  \epsilon +\lambda  \text{trace}(\epsilon) I, \quad \epsilon = \frac{1}{2}(\nabla u + (\nabla u)^\top)
# \end{equation}
# is implemented, and to get the tensor for this law we call:

# In[2]:


# Create stiffness matrix
lam = np.ones(g.num_cells)
mu = np.ones(g.num_cells)
C = pp.FourthOrderTensor(mu, lam)


# Then we need to define boundary conditions. We set the bottom boundary as a Dirichlet boundary, and the other boundaries are set to Neuman.

# In[3]:


# Define boundary type
dirich = np.ravel(np.argwhere(g.face_centers[1] < 1e-10))
bound = pp.BoundaryConditionVectorial(g, dirich, ["dir"] * dirich.size)


# We discretize the stresses by using the multi-point stress approximation (for details, please see: E. Keilegavlen and J. M. Nordbotten. “Finite volume methods for elasticity with weak symmetry”. In: International Journal for Numerical Methods in Engineering (2017)).

# We now define the values we put on the boundaries. We clamp the bottom boundary, and push down by a unitary traction on the top boundary. Note that the value of the Neumann condition given on a face $\pi$ is the integrated traction $\int_\pi g_N d\vec x$, hence the multiplication by face areas.

# In[4]:


top_faces = np.ravel(np.argwhere(g.face_centers[1] > n - 1e-10))
bot_faces = np.ravel(np.argwhere(g.face_centers[1] < 1e-10))

u_b = np.zeros((g.dim, g.num_faces))
u_b[1, top_faces] = -1 * g.face_areas[top_faces]
u_b[:, bot_faces] = 0

u_b = u_b.ravel("F")


# We discretize this system using the `Mpsa` class. We assume zero body forces $f=0$.

# In[5]:


parameter_keyword = "mechanics"

mpsa_class = pp.Mpsa(parameter_keyword)
f = np.zeros(g.dim * g.num_cells)

specified_parameters = {
    "fourth_order_tensor": C,
    "source": f,
    "bc": bound,
    "bc_values": u_b,
}
data = pp.initialize_default_data(g, {}, parameter_keyword, specified_parameters)
mpsa_class.discretize(g, data)
A, b = mpsa_class.assemble_matrix_rhs(g, data)

u = np.linalg.solve(A.A, b)


# The unknowns are ordered so that u[0] and u[1] contain the displacement in the x- and y-direction in cell 0, respectively, u[2] gives the x-displacement in cell 1 etc. Thus we can plot the y component of the displacement by writing:

# In[6]:


pp.plot_grid(g, cell_value=u[1::2], plot_2d=True)


# To understand the inner workings of the discretization, and to recover the traction on the grid faces, some more detail is needed. The mpsa discretization creates two sparse matrices "stress" and "bound_stress". They define the discretization of the cell-face traction:
# \begin{equation}
# T = \text{stress} \cdot u + \text{bound_stress} \cdot u_b
# \end{equation}
# Here, $u$ is a vector of cell center displacement and has length g.dim $*$ g.num_cells. The vector $u_b$ is the boundary condition values. It is the displacement for Dirichlet boundaries and traction for Neumann boundaries and has length g.dim $*$ g.num_faces.

# The global linear system is now formed by momentuum balance on all cells. A row in the discretized system reads
# \begin{equation}
# -\int_{\Omega_k} f dv = \int_{\partial\Omega_k} T(n)dA = [div \cdot \text{stress} \cdot u + div\cdot\text{bound_stress}\cdot u_b]_k,
# \end{equation}
# 
# The call to mpsa_class.assemble_matrix_rhs(), creates the left-hand side matrix $ \text{div} \cdot \text{stress} $, the right-hand side vector, which consists of $\text{f}$ and $-\text{div} \cdot \text{bound_stress}$ (note sign change).
# 
# 

# We can also retrieve the traction on the faces, by first accessing the discretization matrices `stress` and `bound_stress`.

# In[7]:


# Stress discretization
stress = data[pp.DISCRETIZATION_MATRICES][parameter_keyword][
    mpsa_class.stress_matrix_key
]
# Discrete boundary conditions
bound_stress = data[pp.DISCRETIZATION_MATRICES][parameter_keyword][
    mpsa_class.bound_stress_matrix_key
]


T = stress * u + bound_stress * u_b

T2d = np.reshape(T, (g.dim, -1), order="F")
u_b2d = np.reshape(u_b, (g.dim, -1), order="F")
assert np.allclose(np.abs(u_b2d[bound.is_neu]), np.abs(T2d[bound.is_neu]))

T = np.vstack((T2d, np.zeros(g.num_faces)))
pp.plot_grid(g, vector_value=T, figsize=(15, 12), alpha=0)


# Note that the traction on face i: T[2*i:2*i+g.dim] is the traction on the face as defined by the normal vectors `g.face_normals`. This means that for the bottom boundary, the traction T[bot] is the force from the interior of the box to the outside (since the normal vectors here are [0,1]), while on the top boundary, the traction T[top] is the force applied to to top faces from the outside (since the normals here point out of the domain).
# 
# # What we have explored
# We have seen how linear elastic problems can be solved using the Mpsa class, which implements a multi-point finite volume method for momentum balance. The primary unknown in Mpsa is the cell center displacements, whereas tractions on faces can easily be post-processed.
