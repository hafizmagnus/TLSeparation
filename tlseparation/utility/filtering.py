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
from knnsearch import (set_nbrs_knn, set_nbrs_rad)
from data_utils import (get_diff, remove_duplicates)
from shortpath import (array_to_graph, extract_path_info)
from sklearn.neighbors import NearestNeighbors


def continuity_filter(wood, leaf, rad=0.05, n_samples=[]):

    """
    Function to apply a continuity filter to a point cloud that contains gaps
    defined as points from a second point cloud.
    This function works assuming that the continuous variable is the
    wood portion of a tree point cloud and the gaps in it are empty space
    or missclassified leaf data. In this sense, this function tries to correct
    gaps where leaf points are present.

    Parameters
    ----------
    wood: array
        Wood point cloud to be filtered.
    leaf: array
        Leaf point cloud, that might be causing discontinuities in the
        wood point cloud.
    rad: float
        Radius to search for neighboring points in the iterative process.

    Returns
    -------
    wood: array
        Filtered wood point cloud.

    not_wood: array
        Remaining point clouds after the filtering.

    """

    # Stacking wood and leaf arrays.
    arr = np.vstack((wood, leaf))

    # Obtaining wood point cloud indices.
    wood_id = np.arange(wood.shape[0])

    # Calculating shortest path graph over sampled array.
    G = array_to_graph(arr, 0, 3, 100, 0.05, 0.02, 0.5)
    _, dist = extract_path_info(G, 0, return_path=False)

    # Generating nearest neighbors search for the entire point cloud (arr).
    nbrs = NearestNeighbors(algorithm='kd_tree', leaf_size=10,
                            n_jobs=-1).fit(arr)

    # Converting dist variable to array, as it is originaly a list.
    dist = np.asarray(dist)

    # Selecting points and accummulated distance for all wood points in arr.
    gp = arr[wood_id]
    d = dist[wood_id]

    # Preparing control variables to iterate over. idbase will be all initial
    # wood ids and pts all initial wood points. These variables are the ones
    # to use in search of possible missclassified neighbors.
    idbase = wood_id
    pts = gp

    # Setting treshold variables to iterative process.
    e = 9999999
    e_threshold = 3

    # Iterating until threshold is met.
    while e > e_threshold:

        # Obtaining the neighbor indices of current set of points (pts).
        idx2 = nbrs.radius_neighbors(pts, radius=rad,
                                     return_distance=False)

        # Initializing temporary variable id1.
        id1 = []
        # Looping over nn search indices and comparing their respective
        # distances to center point distance. If nearest neighbor distance (to
        # point cloud base) is smaller than center point distance, then ith
        # point is also wood.
        for i in range(idx2.shape[0]):
            for i_ in idx2[i]:
                if dist[i_] <= (d[i]):
                    id1.append(i_)

        # Uniquifying id1.
        id1 = np.unique(id1)

        # Comparing original idbase to new wood ids (id1).
        comp = np.in1d(id1, idbase)

        # Maintaining only new ids for next iteration.
        diff = id1[np.where(~comp)[0]]
        idbase = np.unique(np.hstack((idbase, id1)))

        # Passing new wood points to pts and recalculating e value.
        pts = arr[diff]
        e = pts.shape[0]

        # Passing accummulated distances from new points to d.
        d = dist[diff]

        # Stacking new points to initial wood points and removing duplicates.
        gp = np.vstack((gp, pts))
        gp = remove_duplicates(gp)

    # Removing duplicates from final wood points and obtaining not_wood points
    # from the difference between final wood points and full point cloud.
    wood = remove_duplicates(gp)
    not_wood = get_diff(wood, arr)

    return wood, not_wood


def array_majority(arr_1, arr_2, **kwargs):

    """
    Applies majority filter on two arrays.

    Args:
        arr_1 (array): n-dimensional array of points to filter.
        arr_2 (array): n-dimensional array of points to filter.
        **kwargs: knn, rad.
        knn (int or float): Number neighbors to select around each point in
            arr in order to apply the majority criteria.
        rad (int or float): Search radius arount each point in arr to select
            neighbors in order to apply the majority criteria.

    Returns:
        c_maj_1 (array): Filtered n-dimensional array of the same class as the
            input 'arr_1'.
        c_maj_2 (array): Filtered n-dimensional array of the same class as the
            input 'arr_2'.

    Raises:
        AssertionError: Raised if neither 'knn' or 'rad' arguments are passed
            with valid values (int or float).

    """

    # Asserting input arguments are valid.
    assert ('knn' in kwargs.keys()) or ('rad' in kwargs.keys()), 'Please\
 input a value for either "knn" or "rad".'

    if 'knn' in kwargs.keys():
        assert (type(kwargs['knn']) == int) or (type(kwargs['knn']) ==
                                                float), \
            '"knn" variable must be of type int or float.'
    elif 'rad' in kwargs.keys():
        assert (type(kwargs['rad']) == int) or (type(kwargs['rad']) ==
                                                float), \
            '"rad" variable must be of type int or float.'

    # Stacking the arrays from both classes to generate a combined array.
    arr = np.vstack((arr_1, arr_2))

    # Generating the indices for the local subsets of points around all points
    # in the combined array. Function used is based upon the argument passed.
    if 'knn' in kwargs.keys():
        indices = set_nbrs_knn(arr, arr, kwargs['knn'], return_dist=False)
    elif 'rad' in kwargs.keys():
        indices = set_nbrs_rad(arr, arr, kwargs['rad'], return_dist=False)

    # Making sure indices has type int.
    indices = indices.astype(int)

    # Generating the class arrays from both classified arrays and combining
    # them into a single classes array (classes).
    class_1 = np.full(arr_1.shape[0], 1, dtype=np.int)
    class_2 = np.full(arr_2.shape[0], 2, dtype=np.int)
    classes = np.hstack((class_1, class_2)).T

    # Allocating output variable.
    c_maj = np.zeros(classes.shape)

    # Selecting subset of classes based on the neighborhood expressed by
    # indices.
    class_ = classes[indices]

    # Looping over all points in indices.
    for i in range(len(indices)):

        # Counting the number of occurrences of each value in the ith instance
        # of class_.
        unique, count = np.unique(class_[i, :], return_counts=True)
        # Appending the majority class into the output variable.
        c_maj[i] = unique[np.argmax(count)]

    return arr[c_maj == 1], arr[c_maj == 2]


def class_filter(arr_1, arr_2, target, **kwargs):

    """
    Function to apply class filter on an array based on the combination of
    classed from both arrays (arr_1 and arr_2). Which array gets filtered
    is defined by ''target''.

    Args:
        arr_1 (array): n-dimensional array of points to filter.
        arr_2 (array): n-dimensional array of points to filter.
        target (int or float): number of the input array to filter. Valid
            values are 0 or 1.
        **kwargs: knn, rad.
        knn (int or float): Number neighbors to select around each point in
            arr in order to apply the majority criteria.
        rad (int or float): Search radius arount each point in arr to select
            neighbors in order to apply the majority criteria.

    Returns:
        c_maj_1 (array): Filtered n-dimensional array of the same class as the
            input 'arr_1'.
        c_maj_2 (array): Filtered n-dimensional array of the same class as the
            input 'arr_2'.

    Raises:
        AssertionError: Raised if neither 'knn' or 'rad' arguments are passed
            with valid values (int or float).
        AssertionError: Raised if 'target' variable is not an int or float with
            value 0 or 1.

    """

    # Asserting input arguments are valid.
    assert ('knn' in kwargs.keys()) or ('rad' in kwargs.keys()), 'Please\
 input a value for either "knn" or "rad".'

    if 'knn' in kwargs.keys():
        assert (type(kwargs['knn']) == int) or (type(kwargs['knn']) ==
                                                float), \
            '"knn" variable must be of type int or float.'
    elif 'rad' in kwargs.keys():
        assert (type(kwargs['rad']) == int) or (type(kwargs['rad']) ==
                                                float), \
            '"rad" variable must be of type int or float.'

    assert (type(target) == int) or (type(target) == float), '"target"\
 variable must be of type int or float.'
    assert (target == 0) or (target == 1), '"target" variable must be either\
 0 or 1.'

    # Stacking the arrays from both classes to generate a combined array.
    arr = np.vstack((arr_1, arr_2))

    # Generating the class arrays from both classified arrays and combining
    # them into a single classes array (classes).
    class_1 = np.full(arr_1.shape[0], 1, dtype=np.int)
    class_2 = np.full(arr_2.shape[0], 2, dtype=np.int)
    classes = np.hstack((class_1, class_2)).T

    # Generating the indices for the local subsets of points around all points
    # in the combined array. Function used is based upon the argument passed.
    if 'knn' in kwargs.keys():
        indices = set_nbrs_knn(arr, arr, kwargs['knn'], return_dist=False)
    elif 'rad' in kwargs.keys():
        indices = set_nbrs_rad(arr, arr, kwargs['rad'], return_dist=False)

    # Making sure indices has type int.
    indices = indices.astype(int)

    # Allocating output variable.
    c_maj = classes.copy()

    # Selecting subset of classes based on the neighborhood expressed by
    # indices.
    class_ = classes[indices]

    # Checking for the target class.
    target_idx = np.where(classes == target)[0]

    # Looping over the target points to filter.
    for i in target_idx:

        # Counting the number of occurrences of each value in the ith instance
        # of class_.
        count = np.bincount(class_[i, :])
        # Appending the majority class into the output variable.
        c_maj[i] = count.argmax()

    return arr[c_maj == 1], arr[c_maj == 2]


def dist_majority(arr_1, arr_2, **kwargs):

    """
    Applies majority filter on two arrays.

    Args:
        arr_1 (array): n-dimensional array of points to filter.
        arr_2 (array): n-dimensional array of points to filter.
        **kwargs: knn, rad.
        knn (int or float): Number neighbors to select around each point in
            arr in order to apply the majority criteria.
        rad (int or float): Search radius arount each point in arr to select
            neighbors in order to apply the majority criteria.

    Returns:
        c_maj_1 (array): Filtered n-dimensional array of the same class as the
            input 'arr_1'.
        c_maj_2 (array): Filtered n-dimensional array of the same class as the
            input 'arr_2'.

    Raises:
        AssertionError: Raised if neither 'knn' or 'rad' arguments are passed
            with valid values (int or float).

    """

    # Asserting input arguments are valid.
    assert ('knn' in kwargs.keys()) or ('rad' in kwargs.keys()), 'Please\
 input a value for either "knn" or "rad".'

    if 'knn' in kwargs.keys():
        assert (type(kwargs['knn']) == int) or (type(kwargs['knn']) ==
                                                float), \
            '"knn" variable must be of type int or float.'
    elif 'rad' in kwargs.keys():
        assert (type(kwargs['rad']) == int) or (type(kwargs['rad']) ==
                                                float), \
            '"rad" variable must be of type int or float.'

    # Stacking the arrays from both classes to generate a combined array.
    arr = np.vstack((arr_1, arr_2))

    # Generating the indices for the local subsets of points around all points
    # in the combined array. Function used is based upon the argument passed.
    if 'knn' in kwargs.keys():
        dist, indices = set_nbrs_knn(arr, arr, kwargs['knn'])
    elif 'rad' in kwargs.keys():
        dist, indices = set_nbrs_rad(arr, arr, kwargs['rad'])

    # Making sure indices has type int.
    indices = indices.astype(int)

    # Generating the class arrays from both classified arrays and combining
    # them into a single classes array (classes).
    class_1 = np.full(arr_1.shape[0], 1, dtype=np.int)
    class_2 = np.full(arr_2.shape[0], 2, dtype=np.int)
    classes = np.hstack((class_1, class_2)).T

    # Allocating output variable.
    c_maj = np.zeros(classes.shape)

    # Selecting subset of classes based on the neighborhood expressed by
    # indices.
    class_ = classes[indices]

    # Looping over all points in indices.
    for i in range(len(indices)):

        # Obtaining classe from indices i.
        c = class_[i, :]
        # Caculating accummulated distance for each class.
        d1 = np.sum(dist[i][c == 1])
        d2 = np.sum(dist[i][c == 2])
        # Checking which class has the highest distance and assigning it
        # to current index in c_maj.
        if d1 >= d2:
            c_maj[i] = 1
        elif d1 < d2:
            c_maj[i] = 2

    return arr[c_maj == 1], arr[c_maj == 2]
