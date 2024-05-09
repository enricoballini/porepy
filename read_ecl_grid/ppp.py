import os
import sys
import pdb

import numpy as np
import scipy as sp
from scipy import sparse as sps

"""
read grid in mrst format and return pp.Grid
"""


sys.path.remove("/home/inspiron/Desktop/PhD/porepy/src")
# sys.path.remove("/home/inspiron/.local/lib/python3.10/site-packages/porepy")
sys.path.insert(0, "/home/inspiron/Desktop/PhD/eni_venv/porepy/src")
import porepy as pp

os.system("clear")

np.set_printoptions(threshold=sys.maxsize)

mrst_grid = sp.io.loadmat("mrst_grid.mat")


nodes = mrst_grid["node_coord"].T.astype(np.float64)

fn_row = mrst_grid["fn_node_ind"].astype(np.int32).ravel() - 1
# fn_col = mrst_grid["fn_face_ind"].astype(np.int32).ravel() - 1
# fn_data = np.ones(fn_row.size, dtype=np.int32) # WAS
fn_data = np.ones(fn_row.size, dtype=bool)
# fn = sps.coo_matrix(
#     (fn_data, (fn_row, fn_col)), shape=(fn_row.max() + 1, fn_col.max() + 1)
# ).tocsc() # WAS
indptr = mrst_grid["fn_indptr"].astype(np.int32).ravel() - 1
fn = sps.csc_matrix(
    (fn_data, fn_row, indptr),
    shape=(fn_row.max() + 1, indptr.shape[0] - 1),  # indptr.shape[0] - 1 a caso...
)

# fc_csc = sps.csc_matrix((fn_data, (fn_row, fn_col)))


cf_row = mrst_grid["cf_face_ind"].astype(np.int32).ravel() - 1
cf_col = mrst_grid["cf_cell_ind"].astype(np.int32).ravel() - 1
cf_data = mrst_grid["cf_sgn"].ravel().astype(np.float64)

cf = sps.coo_matrix(
    (cf_data, (cf_row, cf_col)), shape=(cf_row.max() + 1, cf_col.max() + 1)
).tocsc()

dim = nodes.shape[0]

if dim == 2:
    nodes = np.vstack((nodes, np.zeros(nodes.shape[1])))

g = pp.Grid(dim, nodes, fn, cf, "myname")

np.set_printoptions(threshold=sys.maxsize)

# g_pp = pp.CartGrid((3, 2, 2))
# g_pp.compute_geometry()
# # pp.plot_grid(g_pp, alpha=0, info="n")

g.compute_geometry()

print("gonna write vtu")
exporter = pp.Exporter(g, "./grid")
exporter.write_vtu()

print("gonna print with pp.plot_grid")
pp.plot_grid(g)


print("\nDone!")
