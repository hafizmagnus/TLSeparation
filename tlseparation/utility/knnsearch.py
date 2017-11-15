# -*- coding: utf-8 -*-
"""
Module to manage the nearest neighbors selection.

@author: Matheus Boni Vicari (matheus.boni.vicari@gmail.com)
"""

import numpy as np
from sklearn.neighbors import NearestNeighbors


def set_nbrs_rad(arr, pts, rad, return_dist=True):

    """
    Function to create a set of nearest neighbors indices and their respective
    distances for a set of points.

    Parameters
    ----------
    arr: array
        N-dimensional array to perform the knn search on.
    pts: array
        N-dimensional array to search for on the knn search.
    knn: int
        Number of nearest neighbors to search for.
    return_dist: boolean
        Option to return or not the distances of each neighbor.

    Returns
    -------
    indices: array
        Set of neighbors indices from 'arr' for each entry in 'pts'.
    distance: array
        Distances from each neighbor to each central point in 'pts'.

    """

    # Initiating the nearest neighbors search and fitting it to the input
    # array.
    nbrs = NearestNeighbors(radius=rad, metric='euclidean',
                            algorithm='kd_tree', leaf_size=15,
                            n_jobs=-1).fit(arr)

    # Checking if the function should return the distance as well or only the
    # neighborhood indices.
    if return_dist is True:
        # Obtaining the neighborhood indices and their respective distances
        # from the center point.
        distance, indices = nbrs.radius_neighbors(pts)
        return distance, indices

    elif return_dist is False:
        # Obtaining the neighborhood indices only.
        indices = nbrs.radius_neighbors(pts, return_distance=False)
        return indices


def set_nbrs_knn_block(arr, pts, knn, return_dist=True, block_size=100000):

    """
    Function to create a set of nearest neighbors indices and their respective
    distances for a set of points. This function is a variation of
    set_nbrs_knn that sets a limit size for a block of points to query. This
    makes it less efficient in terms of processing time, but avoids running
    out of memory in cases of very dense/large arrays/queries.

    Parameters
    ----------
    arr: array
        N-dimensional array to perform the knn search on.
    pts: array
        N-dimensional array to search for on the knn search.
    knn: int
        Number of nearest neighbors to search for.
    return_dist: boolean
        Option to return or not the distances of each neighbor.
    block_size: int
        Limit of points to query. The variable 'pts' will be subdivided in
        n blocks of size block_size to perform query.

    Returns
    -------
    indices: array
        Set of neighbors indices from 'arr' for each entry in 'pts'.
    distance: array
        Distances from each neighbor to each central point in 'pts'.

    """

    # Initiating the nearest neighbors search and fitting it to the input
    # array.
    nbrs = NearestNeighbors(n_neighbors=knn, metric='euclidean',
                            algorithm='kd_tree', leaf_size=15,
                            n_jobs=-1).fit(arr)

    # Creating block of ids.
    ids = np.arange(pts.shape[0])
    ids = np.array_split(ids, int(pts.shape[0] / block_size))

    # Initializing variables to store distance and indices.
    if return_dist is True:
        distance = np.zeros([pts.shape[0], knn])
    indices = np.zeros([pts.shape[0], knn])

    # Checking if the function should return the distance as well or only the
    # neighborhood indices.
    if return_dist is True:
        # Obtaining the neighborhood indices and their respective distances
        # from the center point by looping over blocks of ids.
        for i in ids:
            nbrs_dist, nbrs_ids = nbrs.kneighbors(pts[i])
            distance[i] = nbrs_dist
            indices[i] = nbrs_ids
        return distance, indices

    elif return_dist is False:
        # Obtaining the neighborhood indices only  by looping over blocks of
        # ids.
        for i in ids:
            nbrs_ids = nbrs.kneighbors(pts[i], return_distance=False)
            indices[i] = nbrs_ids
        return indices


def set_nbrs_knn(arr, pts, knn, return_dist=True):

    """
    Function to create a set of nearest neighbors indices and their respective
    distances for a set of points.

    Parameters
    ----------
    arr: array
        N-dimensional array to perform the knn search on.
    pts: array
        N-dimensional array to search for on the knn search.
    knn: int
        Number of nearest neighbors to search for.
    return_dist: boolean
        Option to return or not the distances of each neighbor.

    Returns
    -------
    indices: array
        Set of neighbors indices from 'arr' for each entry in 'pts'.
    distance: array
        Distances from each neighbor to each central point in 'pts'.

    """

    # Initiating the nearest neighbors search and fitting it to the input
    # array.
    nbrs = NearestNeighbors(n_neighbors=knn, metric='euclidean',
                            algorithm='kd_tree', leaf_size=15,
                            n_jobs=-1).fit(arr)

    # Checking if the function should return the distance as well or only the
    # neighborhood indices.
    if return_dist is True:
        # Obtaining the neighborhood indices and their respective distances
        # from the center point.
        distance, indices = nbrs.kneighbors(pts)
        return distance, indices

    elif return_dist is False:
        # Obtaining the neighborhood indices only.
        indices = nbrs.kneighbors(pts, return_distance=False)
        return indices


def subset_nbrs(distance, indices, new_knn):

    """
    Function to perform a subseting of points from the results of a nearest
    neighbors search.

    Parameters
    ----------
    distance: array
        Distances from each neighbor to each central point in 'pts'.
    indices: array
        Set of neighbors indices from 'arr' for each entry in 'pts'.
    new_knn: int
        Number of neighbors to select from the initial number of neighbors.

    Returns
    -------
    distance: array
        Subset of distances from each neighbor 'indices'.
    indices: array
        Subset of neighbors indices from 'indices'.

    """

    # Returning the subset of neighbors based on a previous nearest
    # neighbors search.
    return distance[:, :new_knn+1], indices[:, :new_knn+1]
