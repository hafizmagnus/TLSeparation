# Copyright (c) 2017, Matheus Boni Vicari, TLSeparation Project
# All rights reserved.
#
# Redistribution and use in source and binary forms, with or without
# modification, are permitted provided that the following conditions are met:
#
#     1. Redistributions of source code must retain the above copyright notice,
#        this list of conditions and the following disclaimer.
#
#     2. Redistributions in binary form must reproduce the above copyright
#        notice, this list of conditions and the following disclaimer in the
#        documentation and/or other materials provided with the distribution.
#
#     3. Neither the name of the Raysect Project nor the names of its
#        contributors may be used to endorse or promote products derived from
#        this software without specific prior written permission.
#
# THIS SOFTWARE IS PROVIDED BY THE COPYRIGHT HOLDERS AND CONTRIBUTORS "AS IS"
# AND ANY EXPRESS OR IMPLIED WARRANTIES, INCLUDING, BUT NOT LIMITED TO, THE
# IMPLIED WARRANTIES OF MERCHANTABILITY AND FITNESS FOR A PARTICULAR PURPOSE
# ARE DISCLAIMED. IN NO EVENT SHALL THE COPYRIGHT HOLDER OR CONTRIBUTORS BE
# LIABLE FOR ANY DIRECT, INDIRECT, INCIDENTAL, SPECIAL, EXEMPLARY, OR
# CONSEQUENTIAL DAMAGES (INCLUDING, BUT NOT LIMITED TO, PROCUREMENT OF
# SUBSTITUTE GOODS OR SERVICES; LOSS OF USE, DATA, OR PROFITS; OR BUSINESS
# INTERRUPTION) HOWEVER CAUSED AND ON ANY THEORY OF LIABILITY, WHETHER IN
# CONTRACT, STRICT LIABILITY, OR TORT (INCLUDING NEGLIGENCE OR OTHERWISE)
# ARISING IN ANY WAY OUT OF THE USE OF THIS SOFTWARE, EVEN IF ADVISED OF THE
# POSSIBILITY OF SUCH DAMAGE.

__author__ = "Matheus Boni Vicari"
__copyright__ = "Copyright 2017, TLSeparation Project"
__credits__ = ["Matheus Boni Vicari"]
__license__ = "GPL3"
__version__ = "1.2.1.1"
__maintainer__ = "Matheus Boni Vicari"
__email__ = "matheus.boni.vicari@gmail.com"
__status__ = "Development"

import numpy as np
from sklearn.neighbors import NearestNeighbors


def set_nbrs_knn(arr, pts, knn, return_dist=True, block_size=100000):

    """
    Function to create a set of nearest neighbors indices and their respective
    distances for a set of points. This function uses a knn search and sets a
    limit size for a block of points to query. This makes it less efficient in
    terms of processing time, but avoids running out of memory in cases of
    very dense/large arrays/queries.

    Args:
        arr (array): N-dimensional array to perform the knn search on.
        pts (array): N-dimensional array to search for on the knn search.
        knn (int): Number of nearest neighbors to search for.
        return_dist (boolean): Option to return or not the distances of each
            neighbor.
        block_size (int): Limit of points to query. The variable 'pts' will be
            subdivided in n blocks of size block_size to perform query.

    Returns:
        indices (array): Set of neighbors indices from 'arr' for each entry in
            'pts'.
        distance (array): Distances from each neighbor to each central point
            in 'pts'.

    """

    # Initiating the nearest neighbors search and fitting it to the input
    # array.
    nbrs = NearestNeighbors(n_neighbors=knn, metric='euclidean',
                            algorithm='kd_tree', leaf_size=15,
                            n_jobs=-1).fit(arr)

    # Making sure block_size is limited by at most the number of points in
    # arr.
    if block_size > pts.shape[0]:
        block_size = pts.shape[0]

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


def set_nbrs_rad(arr, pts, rad, return_dist=True, block_size=100000):

    """
    Function to create a set of nearest neighbors indices and their respective
    distances for a set of points. This function uses a radius search and sets
    a limit size for a block of points to query. This makes it less efficient
    in terms of processing time, but avoids running out of memory in cases of
    very dense/large arrays/queries.

    Args:
        arr (array): N-dimensional array to perform the radius search on.
        pts (array): N-dimensional array to search for on the knn search.
        rad (float): Radius of the NearestNeighbors search.
        return_dist (boolean): Option to return or not the distances of each
            neighbor.
        block_size (int): Limit of points to query. The variable 'pts' will be
            subdivided in n blocks of size block_size to perform query.

    Returns:
        indices (array): Set of neighbors indices from 'arr' for each entry in
            'pts'.
        distance (array): Distances from each neighbor to each central point
            in 'pts'.

    """

    # Making sure block_size is limited by at most the number of points in
    # arr.
    if block_size > pts.shape[0]:
        block_size = pts.shape[0]

    # Initiating the nearest neighbors search and fitting it to the input
    # array.
    nbrs = NearestNeighbors(radius=rad, metric='euclidean',
                            algorithm='kd_tree', leaf_size=15,
                            n_jobs=-1).fit(arr)

    # Creating block of ids.
    ids = np.arange(pts.shape[0])
    ids = np.array_split(ids, int(pts.shape[0] / block_size))

    # Initializing variables to store distance and indices.
    if return_dist is True:
        distance = []
    indices = []

    # Checking if the function should return the distance as well or only the
    # neighborhood indices.
    if return_dist is True:
        # Obtaining the neighborhood indices and their respective distances
        # from the center point by looping over blocks of ids.
        for i in ids:
            nbrs_dist, nbrs_ids = nbrs.radius_neighbors(pts[i])
            for j in i:
                distance.append(nbrs_dist[j])
                indices.append(nbrs_ids[j])
        return distance, indices

    elif return_dist is False:
        # Obtaining the neighborhood indices only  by looping over blocks of
        # ids.
        for i in ids:
            nbrs_ids = nbrs.radius_neighbors(pts[i], return_distance=False)
            for j in i:
                indices.append(nbrs_ids[j])
        return indices


def subset_nbrs(distance, indices, new_knn):

    """
    Performs a subseting of points from the results of a nearest neighbors
    search.

    Args:
        distance (array): Distances from each neighbor to each central point
            in 'pts'.
        indices (array): Set of neighbors indices from 'arr' for each entry in
            'pts'.
        new_knn (array): Number of neighbors to select from the initial number
            of neighbors.

    Returns:
        distance (array): Subset of distances from each neighbor 'indices'.
        indices (array): Subset of neighbors indices from 'indices'.

    """

    # Initializing new_distance and new_indices variables.
    new_distance = []
    new_indices = []

    # Looping over each sample in distance and indices.
    for d, i in zip(distance, indices):
        # Checks if new knn values are smaller than current distance and
        # indices rows. This avoids errors of trying to select a number of
        # columns larger than the available columns.
        if distance.shape[1] >= new_knn:
            new_distance.append(d[:new_knn+1])
            new_indices.append(i[:new_knn+1])
        else:
            new_distance.append(d)
            new_indices.append(i)

    # Returning new_distance and new_indices as arrays.
    return np.asarray(new_distance), np.asarray(new_indices)
