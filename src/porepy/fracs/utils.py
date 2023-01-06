""" Frontend utility functions related to fractures and their meshing.

"""
import logging

import numpy as np

import porepy as pp
from porepy import LineFracture

# Module level logger
logger = logging.getLogger(__name__)


def fracture_length_2d(pts, edges):
    """Find the length of 2D fracture traces.

    Args:
        pts (np.ndarray, 2 x n_pts): Coordinates of start and endpoints of
            fractures.
        edges (np.ndarary, 2 x n_fracs): Indices of start and endpoint of
            fractures, referring to columns in pts.

    Returns:
        np.ndarray, length n_fracs: Length of each fracture.

    """
    start = pts[:, edges[0]]
    end = pts[:, edges[1]]

    length = np.sqrt(np.sum(np.power(end - start, 2), axis=0))
    return length


def uniquify_points(pts, edges, tol):
    """Uniquify a set of points by merging almost coinciding coordinates.

    Also update fractures, and remove edges that consist of a single point
    (either after the points were merged, or because the input was a point
    edge).

    Args:
        pts (np.ndarary, n_dim x n_pts): Coordinates of start and endpoints of
            the fractures.
        edges (np.ndarray, n x n_fracs): Indices of start and endpoint of
            fractures, referring to columns in pts. Should contain at least two
            rows; additional rows representing fracture tags are also accepted.
        tol (double): Tolerance used for merging points.

    Returns:
        np.ndarray (n_dim x n_pts_unique): Unique point array.
        np.ndarray (2 x n_fracs_update): Updated start and endpoints of
            fractures.
        np.ndarray: Index (referring to input) of fractures deleted as they
            effectively contained a single coordinate.

    """

    # uniquify points based on coordinates
    p_unique, _, o2n = pp.utils.setmembership.unique_columns_tol(pts, tol=tol)
    # update edges
    e_unique_p = np.vstack((o2n[edges[:2]], edges[2:]))

    # Find edges that start and end in the same point, and delete them
    point_edge = np.where(np.diff(e_unique_p[:2], axis=0)[0] == 0)[0].ravel()
    e_unique = np.delete(e_unique_p, point_edge, axis=1)

    return p_unique, e_unique, point_edge


def linefractures_to_pts_edges(
    fractures: list[LineFracture],
) -> tuple[np.ndarray, np.ndarray]:
    """Convert a list of fractures into two numpy arrays of the corresponding points and
    edges.

    Parameters:
        fractures: List of fractures.

    Returns:
        pts ``(shape=(2, np))``: Coordinates of the start- and endpoints of the
        fractures.
        edges ``(shape=(len(fractures), 2), dtype=int)``: Indices for the start- and
        endpoint of each fracture. Note, that one point in ``pts`` may be the start-
        and/or endpoint of multiple fractures.

    """
    pts_list: list[np.ndarray] = []
    edges_list: list[list[int]] = []
    for frac in fractures:
        edge = []
        for point in frac.points():
            compare_points = [np.allclose(point, x) for x in pts_list]
            if not any(compare_points):
                pts_list.append(point)
                edge.append(len(pts_list) - 1)
            else:
                edge.append(compare_points.index(True))
        edges_list.append(edge)
    pts = np.array(pts_list).squeeze().T
    edges = np.array(edges_list, dtype=int).T
    return pts, edges


def pts_edges_to_linefractures(
    pts: np.ndarray, edges: np.ndarray
) -> list[LineFracture]:
    """Convert points and edges into a list of fractures

    Parameters:
        pts ``(shape=(2, np))``: _description_
        edges ``(shape=(len(fractures), 2), dtype=int)``: _description_

    Returns:
        List of fractures.
    """
    fractures: list[LineFracture] = []
    for start_index, end_index in zip(edges[0, :], edges[1, :]):
        fractures.append(
            LineFracture(np.array([pts[:, start_index], pts[:, end_index]]).T)
        )
    return fractures
