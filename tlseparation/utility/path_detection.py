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
import networkx as nx
from data_utils import remove_duplicates
from shortpath import create_graph_iter
from shortpath import base_center
from sklearn.neighbors import NearestNeighbors
from collections import defaultdict


def cont_filt(wood, leaf, dist_threshold=0.05):

    arr = np.vstack((wood, leaf))
    wood_ids = np.arange(wood.shape[0])

    center = base_center(arr, base_length=0.3)[0]

    G = create_graph_iter(arr, n_neighbors=5, nn_step=2,
                          dist_threshold=np.inf, maxiter=20)

    print('Graph created')
    nbrs = NearestNeighbors(leaf_size=15, n_jobs=-1).fit(arr)
    base_id = nbrs.kneighbors(center.reshape(1, -1), 1,
                              return_distance=False)[0][0]

    mask = np.zeros(G.number_of_nodes()).astype(bool)

    # Calculating the shortest path
    shortpath = nx.single_source_dijkstra_path_length(G, base_id)
    # Obtaining the node coordinates and their respective distance from
    # the base point.
    nodes_ids = shortpath.keys()
    dist = shortpath.values()
    # Obtaining path list for every node.
    path = nx.single_source_dijkstra_path(G, base_id)
    # Obtaining nodes coordinates.
    nodes = arr[nodes_ids]

    dist = np.array(dist)

    path = path.values()
    path_nodes = [i for j in path for i in j]

    mask[path_nodes] = True
    mask[wood_ids] = True

    e = np.inf
    threshold = 1

    print('Starting region growing')
    while e > threshold:
        nbrs = NearestNeighbors(leaf_size=15, n_jobs=-1).fit(arr[~mask])
        e1 = np.sum(mask)

        nbrs_dist, nbrs_ids = nbrs.kneighbors(arr[mask], 1)

        for i, nbr_i in enumerate(nbrs_ids[nbrs_dist <= dist_threshold]):
            if dist[nbr_i] <= dist[mask][i]:
                mask[nbr_i] = True

        e2 = np.sum(mask)
        e = e2 - e1
#        e = nbrs_ids.shape[0]
        print e

    return nodes[mask], nodes[~mask]


def path_freq(arr, voxdim, freq_threshold):

    vox = np.array(arr / voxdim).astype(int)

    voxels = defaultdict(list)
    for i, v in enumerate(vox):
        voxels[tuple(v)].append(i)

    vox_uniques = remove_duplicates(vox)
    vox_uniques = vox_uniques * voxdim
#    vox_uids = np.arange(vox_uniques.shape[0])

    center = base_center(arr, base_length=0.3)[0]

    G = create_graph_iter(vox_uniques, n_neighbors=5, nn_step=2,
                          dist_threshold=np.inf, maxiter=20)

    print('Graph created')
    nbrs = NearestNeighbors(leaf_size=15, n_jobs=-1).fit(vox_uniques)
    base_id = nbrs.kneighbors(center.reshape(1, -1), 1,
                              return_distance=False)[0][0]

    mask = np.zeros(G.number_of_nodes())

    # Calculating the shortest path
    shortpath = nx.single_source_dijkstra_path_length(G, base_id)
    # Obtaining the node coordinates and their respective distance from
    # the base point.
    nodes_ids = shortpath.keys()
    dist = shortpath.values()
    # Obtaining path list for every node.
    path = nx.single_source_dijkstra_path(G, base_id)
    # Obtaining nodes coordinates.
    nodes = vox_uniques[nodes_ids]

    dist = np.array(dist)

    path = path.values()
    path_nodes = [i for j in path for i in j]

    # Obtaining all unique values in the central nodes path and their
    # respective frequency.
    path_nodes, freq = np.unique(path_nodes, return_counts=True)

    # Log transforming the frequency values.
    freq_log = np.log(freq)

    # Filtering the central nodes based on the frequency of paths
    # that contains each node.
    freq_mask = (freq_log >= (np.max(freq_log) * freq_threshold)).astype(bool)
    p = nodes[freq_mask]
    pdist = dist[freq_mask]

    nbrs = NearestNeighbors(leaf_size=15, n_jobs=-1).fit(nodes)
    nbrs_ids = nbrs.radius_neighbors(p, radius=voxdim*3,
                                     return_distance=False)

    mask[freq_mask.astype(bool)] = 1
    for p_id, idx in enumerate(nbrs_ids):
        for id_ in idx:
            if dist[id_] <= pdist[p_id]:
                mask[id_] = 1

    mask = mask.astype(bool)

    e = np.inf
    threshold = 1
    dist_threshold = voxdim*4

    print('Starting region growing')
    while e > threshold:
        nbrs = NearestNeighbors(leaf_size=15,
                                n_jobs=-1).fit(vox_uniques[~mask])
        e1 = np.sum(mask)

        nbrs_dist, nbrs_ids = nbrs.kneighbors(vox_uniques[mask], 1)

        for i, nbr_i in enumerate(nbrs_ids[nbrs_dist <= dist_threshold]):
            if dist[nbr_i] <= dist[mask][i]:
                mask[nbr_i] = True

        e2 = np.sum(mask)
        e = e2 - e1
#        e = nbrs_ids.shape[0]
        print e

    vids = []
    new_voxels = (vox_uniques[mask] / voxdim).astype(int)
    for i in new_voxels:
        vids.append(voxels[tuple(i)])
    voxels_ids = np.unique([i for j in vids for i in j])

    return arr[voxels_ids]


def base_circle_points(center, radius, number_points):

    # Calculating the angle interval between the points.
    angle = (2 * np.pi) / number_points
    # Creating an array of angular values to place points at.
    R = np.arange(0, 2 * np.pi, angle)

    # Calculating the x and y coordinates of a horizontal transect
    # of the shape.
    x = center[0] + (radius * np.cos(R))
    y = center[1] + (radius * np.sin(R))

    return np.vstack((x, y, np.full(x.shape, center[2]))).T
