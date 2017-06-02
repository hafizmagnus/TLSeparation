# -*- coding: utf-8 -*-
"""
@author: Matheus Boni Vicari (matheus.boni.vicari@gmail.com)
"""

import numpy as np
from shortpath import calculate_path
from sklearn.neighbors import NearestNeighbors
import scipy.cluster.hierarchy as hcluster
from point_compare import get_diff
from shortpath_mod import calculate_path_mixed
import hdbscan


def path_clustering(arr, knn, slice_length, cluster_threshold,
                    freq_threshold=0.6):

    """
    Function to generate the path clustering of a point cloud with a defined
    root point.

    Parameters
    ----------
    arr: array_like
        N-dimensional array (m x n) containing a set of parameters (n) over
        a set of observations (m). In this case, the set of parameters are the
        point cloud coordinates, where each row represents a point.
    knn: int
        Number of nearest neighbors to use in the reconstruction of point cloud
        around the generated path nodes. A high value may lead to unnecessary
        duplication of steps. A low value may lead to gaps in the reconstructed
        point cloud.
    slice_length: float
        Length of the slices of the data in 'arr'.
    cluster_threshold: float
        Distance threshold to be used as constraint in the slice clustering
        step.

    Returns
    -------
    wood: array
        N-dimensional array (w x n) containing the (w) points classified as
        wood from the path reconstruction. The columns (n) represents the
        3D coordinates of each point.
    leaf: array
        N-dimensional array (l x n) containing the (l) points not classified as
        wood and, therefore, classified as leaf. The columns (n) represents the
        3D coordinates of each point.

    """

    # Slicing and clustering the data to generate the center points of
    # every cluster. Return also the cluster data and the diameter of
    # each cluster.
    nodes_data, nodes_diameter, nodes, dist  = slice_nodes(arr, slice_length,
                                                            cluster_threshold)

    # Obtaining the central nodes coordinates (tree skeleton points).
    central_nodes = np.asarray(nodes_data.keys())

    # Calculating the shortest path over the central nodes.
    gnodes, gdist, gpath = calculate_path(central_nodes, 'knn', 10)
    gdist = np.array(gdist)

    # Extracting all the nodes in the shortest path.
    gpath = gpath.values()
    gpath_nodes = [i for j in gpath for i in j]

    # Obtaining all unique values in the central nodes path and their
    # respective frequency.
    gpath_nodes, freq = np.unique(gpath_nodes, return_counts=True)

    # Log transforming the frequency values.
    freq_log = np.log(freq)

    # Filtering the central nodes based on the frequency of paths
    # that contains each node.
    gp = gnodes[freq_log >= (np.max(freq_log) * freq_threshold)]
#    gpdist = gdist[freq_log >= (np.max(freq_log) * freq_threshold)]

    # Obtaining list of close nodes that are not yet in 'gp' and stacking them
    # to 'gp'. This step aims to fill the gaps between nodes from 'gp'.
    nbrs = NearestNeighbors(leaf_size=15, n_jobs=-1)
    nbrs.fit(gnodes)

    idx = nbrs.kneighbors(gp, n_neighbors=knn, return_distance=False)
    idx = np.unique(idx)

    gp = np.vstack((gp, gnodes[idx]))
    pts = gp

    npw = gp.shape[0]

    e = 9999999
    e_threshold = 10

    while e > e_threshold:
        idx = nbrs.radius_neighbors(pts, radius=0.06,
                                    return_distance=False)

        id1 = []
        for i in idx:
            id1.append(i[1:][gdist[i[1:]] <= gdist[i[0]]])

        id1 = np.unique([j for i in id1 for j in i])

        pts = get_diff(gp, gnodes[id1])
        gp = np.vstack((gp, pts))

        e = gp.shape[0] - npw
        npw = gp.shape[0]

    # Obtaining the data from each respective node in 'gp'. In this case the
    # data from the skeleton nodes are considered as wood and the remaining
    # data is set as leaf.
    try:
        keys = tuple(map(tuple, gp))
        vals = map(nodes_data.get, keys)
        vals = filter(lambda v: v is not None, vals)
        wood = np.concatenate(vals, axis=0)

        leaf = get_diff(arr, wood)

        return wood, leaf
    except:
        return [], []


def slice_nodes(arr, slice_length, cluster_threshold):

    """
    Function to slice a 3D point cloud by distance from the base and generate a
    skeleton of points.


    Parameters
    ----------
    arr: array_like
        N-dimensional array (m x n) containing a set of parameters (n) over
        a set of observations (m). In this case, the set of parameters are the
        point cloud coordinates, where each row represents a point.
    slice_length: float
        Length of the slices of the data in 'arr'.
    cluster_threshold: float
        Distance threshold to be used as constraint in the slice clustering
        step.

    Returns
    -------
    cluster_data: dict
        Dictionary containing the skeleton nodes coordinates (keys) and the
        substet of points from 'arr' that generated the respective skeleton
        nodes.
    cluster_diameter: dict
        Dictionary containing the skeleton nodes coordinates (keys) and the
        mean diameter of the cluster that generated the respective skeleton
        nodes.

    """

    # Calculating the shortest path distance for the input array (arr).
    # Here, the calculate_path module is called twice in order to reach
    # as many points in 'arr' as possible.
    nodes, dist = calculate_path_mixed(arr, n_neighbors=3, return_path=False)

    # Initializing the dictionary variables for output.
    cluster_data = dict()
    cluster_diameter = dict()

    # Generating the indices of each slice.
    slice_id = np.round((dist - np.min(dist, axis=0)) /
                        slice_length).astype(int)

    # Looping over each slice of the data.
    for i in np.unique(slice_id):
        # Selecting the data from the current slice.
        data_slice = nodes[i == slice_id]

        try:
            # Clustering the data.
            clusters = data_clustering(data_slice, cluster_threshold)
            # Looping over every cluster in the slice.
            for j in np.unique(clusters):
                # Selecting data from the current cluster.
                d = data_slice[j == clusters]

                # Calculating the central coord and diameter of the current
                # cluster.
                center, diameter = central_coord(d)
                xm, ym, zm = center
                cluster_data[xm, ym, zm] = d
                cluster_diameter[xm, ym, zm] = diameter

        except:
            pass

    return cluster_data, cluster_diameter, nodes, dist


def central_coord(arr):

    """
    Function to calculate the central coordinate and the mean diameter of an
    array of points in 3D space.

    Parameters
    ----------
    arr: array_like
        N-dimensional array (m x n) containing a set of parameters (n) over
        a set of observations (m). In this case, the set of parameters are the
        point cloud coordinates, where each row represents a point.

    Returns
    -------
    coord: array_like
        Central coordinates of the points in the input array.
    diameter: float
        Mean diameter of the points in the input array.

    """

    min_ = np.min(arr, axis=0)
    max_ = np.max(arr, axis=0)

    return min_ + ((max_ - min_) / 2), np.mean(max_[:2] - min_[:2])


#def data_clustering(point_arr, threshold):
#
#    """
#    Function to cluster the array slices using hierarchical clustering.
#
#    Parameters
#    ----------
#    point_arr: array
#        N-dimensional array (m x n) containing a set of parameters (n) over
#        a set of observations (m). In this case, the set of parameters are the
#        point cloud coordinates, where each row represents a point.
#    threshold: float
#        Distance threshold to be used as constraint in the slice clustering
#        step.
#
#    Returns
#    -------
#    clusters: array_like
#        Set of cluster labels for the classified array.
#
#    """
#
#    clusters = hcluster.fclusterdata(point_arr, threshold, method='single',
#                                     criterion="distance")
#
#    return clusters


def data_clustering(point_arr, threshold):

    """
    Function to cluster the array slices using hierarchical clustering.

    Parameters
    ----------
    point_arr: array
        N-dimensional array (m x n) containing a set of parameters (n) over
        a set of observations (m). In this case, the set of parameters are the
        point cloud coordinates, where each row represents a point.
    threshold: float
        Distance threshold to be used as constraint in the slice clustering
        step.

    Returns
    -------
    clusters: array_like
        Set of cluster labels for the classified array.

    """

#    clusters = hcluster.fclusterdata(point_arr, threshold, method='single',
#                                     criterion="distance")

    clusterer = hdbscan.HDBSCAN()
    clusterer.fit(point_arr)   

    return clusterer.labels_


def entries_to_remove(entries, d):

    """
    Function to remove selected entries (key and respective values) from
    a given dict.
    Based on a reply from the user mattbornski at stackoverflow.

    Parameters
    ----------
    entries: array_like
        Set of entried to be removed.
    d: dict
        Dictionary to applu the entried removal.

    Reference
    ---------
    ..  [1] mattbornski, 2012. http://stackoverflow.com/questions/8995611/\
removing-multiple-keys-from-a-dictionary-safely
    """

    for k in entries:
        d.pop(k, None)
