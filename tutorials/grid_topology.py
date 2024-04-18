#!/usr/bin/env python
# coding: utf-8

# # Grid topology

# In this tutorial we will have a more advanced take on the grid structure, specifically the topological information of the grids. 
# We recommend reading the [grid](./grids.ipynb) tutorial before starting on this tutorial.

# Before tackling the topological information, we will re-use the basic grid we created in the previous grid structure tutorial:
# 

# In[1]:


import numpy as np
import porepy as pp
import inspect

nx = np.array([3, 2])
g = pp.CartGrid(nx)
g.compute_geometry()


# And for visualization purposes we also include the same plot as before:

# In[2]:


cell_id = np.arange(g.num_cells)
pp.plot_grid(g, cell_value=cell_id, info='cfn', alpha=0.5, figsize=(15,12))


# For the rest of this tutorial, we will advice the user to sporadically have a look at this plot. 
# All the face numbers, node numbers, etc. gathered further down are shown nicely in this figure. 
# It can therefore be helpful as a visual guide. 

# # Topological information

# In addition to storing coordinates of cells, faces and nodes, the grid object also keeps track of the relation between them. 
# Specifically, we can access:
# 1. The relation between cells and faces
# 2. The relation between faces and nodes
# 3. The direction of `face_normals`, as in which of the neighboring cells has the normal vector as outwards pointing.
# 
# Note that there is no notion of edges for 3d grids. 
# These are not usually needed for the type of numerical methods that are primarily of interest in `porepy`. 
# The information can still be recovered from the face-node relations, see comments below.
# 
# The topological information is stored in two attributes, `cell_faces` and `face_nodes`. 
# The latter has the simplest interpretation, so we start out with that one:

# In[3]:


g.face_nodes


# We see that the information is stored as a scipy.sparse matrix. 
# From the shape of the matrix, we conclude that the rows represent nodes, while the faces are stored in columns. 
# We can get the nodes for the first face by brute force by writing

# In[4]:


g.face_nodes[:,0].nonzero()[0]


# Similarly, we can also get the faces of a node, for example the sixth (counting from 0) node:

# In[5]:


g.face_nodes[5,:].nonzero()[1]


# The map between cells and faces is stored in the same way, thus the faces of cell 0 are found by

# In[6]:


faces_of_cell_0 = g.cell_faces[:,0].nonzero()[0]
print(faces_of_cell_0)


# `cell_faces` also keeps track of the direction of the normal vector relative to the neighboring cells.
# This is done by storing data as $\pm 1$, or zero if there is no connection between the cells (in contrast, `face_nodes` simply consist of `True` or `False`).

# In[7]:


g.cell_faces[:,0].data


# Compare this with the face normal vectors, which can either be gathered in an array or shown using `pp.plot_grid`:

# In[8]:


g.face_normals[:g.dim, faces_of_cell_0]


# In[9]:


pp.plot_grid(g, info='co', alpha=0.5, figsize=(10, 12))


# As can be seen from both the normal vector array and the figure, we observe that positive data corresponds to the normal vector pointing out of the cell. 
# This is a very useful feature, since it in effect means that the transpose of `g.cell_faces` is the discrete divergence operator for the grid.
# 

# We can obtain the cells of a face in the same way as we obtained the faces of a node. 
# However, we know that there will be either 1 or 2 cells adjacent to each face. 
# It is thus feasible to create a dense representation of the cell-face relations:

# In[10]:


g.cell_face_as_dense()


# Here, each column represents the cells of a face, and a negative value signifies that the face is on the boundary, i.e., the face has only one neighboring cell. 
# The cells are ordered so that the normal vector points from the cell in row 0 to that in row 1.
# 
# Finally, we note that to get a cell-node relation, we can combine `cell_faces` and `face_nodes`. 
# However, since `cell_faces` contains both positive and negative values, we need to take the absolute value of the data. This procedure is implemented in the method `cell_nodes()`, which returns a sparse matrix that can be handled in the usual way

# In[11]:


print(inspect.getsource(g.cell_nodes))
g.cell_nodes()[:,0].nonzero()[0]


# # What we have explored
# 
# We have seen how to access various topological properties of grids in PorePy, such as the relation between cells, faces and nodes, as well as the direction of the face normal vectors.
