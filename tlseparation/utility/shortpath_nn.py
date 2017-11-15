# -*- coding: utf-8 -*-
"""
Module to manage the shortest path calculation on a point cloud.

@author: Matheus Boni Vicari (matheus.boni.vicari@gmail.com)
"""

import networkx as nx
import numpy as np
from sklearn.neighbors import NearestNeighbors
import itertools
from point_compare import get_diff


def calculate_path_mixed_nn(arr, n_neighbors=5, base=[], nn_step=2,
                            dist_threshold=np.inf, return_path=True,
                            maxiter=20):

    """
    Function to calculate the shortest path of all reachable points in an
    array to a base point. This function will use a combined approach to
    make sure that as many points as possible are converted to the graph
    and, therefore, evaluated.

    Parameters
    ----------
    arr: array
        N-dimensional array (m x n) containing a set of parameters (n) over
        a set of observations (m) to apply the shortest path on.
    n_neighbors: int
        Number of nearest neighbors to search for in order to connect all
        entries in arr to form a graph.
    base: array_like
        N-dimensional array to use as base point coordinates for the shortest
        path calculation.
    dist_threshold: float
        Parameter to limit the creation of an edge longer than a distance
        threshold.
    return_path: boolean
        Option to select if the function should return the indices of entries
        from 'arr' that are part of the shortest paths.

    Returns
    -------
    nodes: array
        N-dimensional array (m x n) of nodes from the shortest path containing
        a set of parameters (n) over a set of observations (m).
    distance: array_like
        Accumulated distance from the base for each node.
    Path: nested_lists
        Lists of indices of entried from 'arr' that are part of the paths for
        each node.

    """

    G = create_graph_iter(arr, n_neighbors=n_neighbors, base=[],
                          nn_step=nn_step, dist_threshold=dist_threshold,
                          maxiter=maxiter)

    if return_path is True:
        nodes_ids, distance, path_list = extract_path_info(G, arr, 0)
        nodes = arr[nodes_ids]
        return nodes, nodes_ids, distance, path_list

    elif return_path is False:
        nodes_ids, distance = extract_path_info(G, arr, 0, return_path=False)
        nodes = arr[nodes_ids]
        return nodes, nodes_ids, distance


def extract_path_info(G, arr, base_id, return_path=True):

    # Calculating the shortest path
    shortpath = nx.single_source_dijkstra_path_length(G, 0)

    # Obtaining the node coordinates and their respective distance from
    # the base point.
    nodes_ids = shortpath.keys()
    distance = shortpath.values()

    # Checking if the function should also return the paths of each node and
    # if so, generating the path list and returning it.
    if return_path is True:
        path_list = nx.single_source_dijkstra_path(G, 0)
        return nodes_ids, distance, path_list

    elif return_path is False:
        return nodes_ids, distance


def create_graph_iter(arr, n_neighbors=5, base=[], nn_step=2,
                      dist_threshold=np.inf, maxiter=20):

    # Checking for the existance of a base point, otherwise, create it.
    if len(base) > 1:
        arr = np.vstack((base, arr))

    arr_rem = arr

    # Initializing Graph.
    G = nx.Graph()

    iter_ = 0
    while arr_rem.shape[0] > 0 and iter_ <= maxiter:

        print iter_
        print arr_rem.shape[0]
        if arr_rem.shape[0] > 0:

            G = create_graph_simple(G, arr, arr_rem, n_neighbors,
                                    dist_threshold)

            n_ids, _ = extract_path_info(G, arr, 0, return_path=False)
            n = arr[n_ids]

            arr_rem = get_diff(n, arr)

            n_neighbors = n_neighbors + nn_step

        iter_ += 1

    return G


def create_graph_simple(G, arr, arr_rem, n_neighbors, threshold):

    """
    Function to create a Graph using the knn search approach.

    Parameters
    ----------
    arr: array
        N-dimensional array (m x n) containing a set of parameters (n) over
        a set of observations (m) to apply the shortest path on.
    n_neighbors: int
        Number of nearest neighbors to search for in order to connect all
        entries in arr to form a graph.
    threshold: float
        Parameter to limit the creation of an edge longer than a distance
        threshold.

    Returns
    -------
    G: graph
        Networkx graph containing the points in 'arr' as nodes and the distance
        between nodes as weight.

    """

    # Initializing the nearest neihbors search and fitting it to the input
    # array.
    nbrs = NearestNeighbors(n_neighbors=n_neighbors, metric='euclidean',
                            algorithm='kd_tree', leaf_size=15,
                            n_jobs=-1).fit(arr)
    # Obtaining the set of indices and distances from the nn search.
    distances, indices = nbrs.kneighbors(arr_rem)

    # Looping over the combination of indices and distances:
    for i, d in itertools.izip(indices, distances):
        # Looping over all neighbors of the current ith central point.
        for c in np.arange(len(indices[0, 1:])):
            if d[c] <= threshold:
                # If the distance between vertices is less than a given
                # threshold, add edge (i[0], i[c]) to Graph.
                G.add_weighted_edges_from([(i[0], i[c], d[c])])

    return G


def base_center(arr, base_length=0.3):

    """
    Function to generate a base center coordinates from an array.

    Parameters
    ----------
    arr: array
        N-dimensional array (m x n) containing a set of parameters (n) over
        a set of observations (m) to use in the creation of a base point
        coordiante.

    Returns
    -------
    coord: array_like
        N-dimensional set of coordinates for the base point from 'arr'.

    """

    # Obtaining the lowest points in arr.
    base_height = np.min(arr[:, 2]) + base_length
    mask = arr[:, 2] <= base_height
    base_pts = arr[mask, :]

    # Calculating the base coordinates.
    base_x = ((np.max(base_pts[:, 0]) + np.min(base_pts[:, 0])) / 2)
    base_y = ((np.max(base_pts[:, 1]) + np.min(base_pts[:, 1])) / 2)
    base_z = np.min(base_pts[:, 2])

    return np.array([base_x, base_y, base_z], ndmin=2)
