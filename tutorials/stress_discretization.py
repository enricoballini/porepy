import os
import pdb

import numpy as np
import porepy as pp

n = 5
g = pp.CartGrid([n, n])
g.compute_geometry()

lam = np.ones(g.num_cells)
mu = np.ones(g.num_cells)
C = pp.FourthOrderTensor(mu, lam)

dirich = np.ravel(np.argwhere(g.face_centers[1] < 1e-10))
bound = pp.BoundaryConditionVectorial(g, dirich, ["dir"] * dirich.size)

top_faces = np.ravel(np.argwhere(g.face_centers[1] > n - 1e-10))
bot_faces = np.ravel(np.argwhere(g.face_centers[1] < 1e-10))

u_b = np.zeros((g.dim, g.num_faces))
u_b[1, top_faces] = -1 * g.face_areas[top_faces]
u_b[:, bot_faces] = 0

u_b = u_b.ravel("F")

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

pp.plot_grid(g, cell_value=u[1::2], plot_2d=True)

stress = data[pp.DISCRETIZATION_MATRICES][parameter_keyword][
    mpsa_class.stress_matrix_key
]
bound_stress = data[pp.DISCRETIZATION_MATRICES][parameter_keyword][
    mpsa_class.bound_stress_matrix_key
]

T = stress * u + bound_stress * u_b

T2d = np.reshape(T, (g.dim, -1), order="F")
u_b2d = np.reshape(u_b, (g.dim, -1), order="F")
assert np.allclose(np.abs(u_b2d[bound.is_neu]), np.abs(T2d[bound.is_neu]))

T = np.vstack((T2d, np.zeros(g.num_faces)))
pp.plot_grid(g, vector_value=T, figsize=(15, 12), alpha=0)
