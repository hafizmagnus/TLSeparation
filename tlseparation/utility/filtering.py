# -*- coding: utf-8 -*-
"""
Module to manage the classification filtering.

@author: Matheus Boni Vicari (matheus.boni.vicari@gmail.com)
"""

import numpy as np
import pandas as pd
from point_compare import get_diff
from knnsearch import set_nbrs_knn
from knnsearch import set_nbrs_rad
from shortpath_nn import calculate_path_mixed_nn
from sklearn.neighbors import NearestNeighbors


def dist_majority_knn(arr_1, arr_2, knn):

    """
    Function to apply majority filter on two arrays.

    Parameters
    ----------
    arr_1: array
        n-dimensional array of points to filter.
    arr_2: array
        n-dimensional array of points to filter.
    knn: int
        Number neighbors to select the subset of points to apply the
        majority criteria.

    Returns
    -------
    c_maj_1: array
        Filtered n-dimensional array of the same class as the input 'arr_1'.
    c_maj_2: array
        Filtered n-dimensional array of the same class as the input 'arr_2'.

    """
    # Stacking the arrays from both classes to generate a combined array.
    arr = np.vstack((arr_1, arr_2))

    # Generating the indices for the local subsets of points around all points
    # in the combined array.
    dist, indices = set_nbrs_knn(arr, arr, knn)

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
    n_samples:
        Number of samples to calculate the shortest path procedure and the
        upscale to the whole point cloud (wood + leaf).

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

    # Checking the number of samples and if not declared, calculate a
    # reasonable number (one third of total points).
    if len(n_samples) < 1:
        n_samples = np.int(arr.shape[0] / 3)

    # Selecting n_samples random indices for sampling arr.
    s = np.random.choice(np.arange(arr.shape[0]), n_samples,
                         replace=False)
    # Get sample array.
    arr_sample = arr[s]

    # Calculating shortest path graph over sampled array.
#    nodes, dist, path = calculate_path_mixed_nn(arr_sample, n_neighbors=3)
    nodes, nodes_ids, dist = calculate_path_mixed_nn(arr_sample, n_neighbors=3,
                                                     return_path=False)

    # Generating nearest neighbors search for the entire point cloud (arr).
    nbrs = NearestNeighbors(algorithm='kd_tree', leaf_size=10,
                            n_jobs=-1).fit(arr)

    # Converting dist variable to array, as it is originaly a list.
    dist = np.asarray(dist)

    # Upscaling dist to entire point cloud (arr).
    dist = apply_nn_value(nodes, arr, dist)

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


def remove_duplicates(arr):

    """
    Function to remove duplicated rows from an array.

    Parameters
    ----------
    arr: array
        N-dimensional array (m x n) containing a set of parameters (n) over a
        set of observations (m).

    Returns
    -------
    unique: array
        N-dimensional array (m* x n) containing a set of unique parameters (n)
        over a set of unique observations (m*).

    """

    # Setting the pandas.DataFrame from the array (arr) data.
    df = pd.DataFrame({'x': arr[:, 0], 'y': arr[:, 1], 'z': arr[:, 2]})

    # Using the drop_duplicates function to remove the duplicate points from
    # df.
    unique = df.drop_duplicates(['x', 'y', 'z'])

    return np.asarray(unique)


def apply_nn_value(base, arr, attr):

    """
    Fundtion to upscale a set of attributes from a base array to another
    denser array.

    Parameters
    ----------
    base: array
        Base array to which the attributes to upscale were originaly matched.
    arr: array
        Target array to which the attributes will be upscaled.
    attr: array
        Attributes to upscale.

    Returns
    -------
    new_attr: array
        Upscales attributes.

    """

    nbrs = NearestNeighbors(n_neighbors=1, algorithm='kd_tree',
                            leaf_size=10, n_jobs=-1).fit(base)
    idx = nbrs.kneighbors(arr, return_distance=False)

    newattr = attr[idx]

    return np.reshape(newattr, newattr.shape[0])


def majority(classes, indices):

    """
    Function to apply a majority filter on a set of classes.

    Parameters
    ----------
    classes: array_like
        1D set of classes labels to apply the filter.
    indices:
        Nearest neihbors indices for each entry in classes.

    Returns
    -------
    c_maj: array_like
        1D set of filtered classes labels.

    """

    # Allocating output variable.
    c_maj = np.zeros(classes.shape)

    # Selecting subset of classes based on the neighborhood expressed by
    # indices.
    class_ = classes[indices]

    # Looping over the target points to filter.
    for i in range(len(indices)):

        # Counting the number of occurrences of each value in the ith instance
        # of class_.
        count = np.bincount(class_[i, :])
        # Appending the majority class into the output variable.
        c_maj[i] = count.argmax()

    return c_maj


def class_filter(arr_1, arr_2, knn, target):

    """
    Function to apply a majority filter on a set of classes, while focusing on
    a class.

    Parameters
    ----------
    classes: array_like
        1D set of classes labels to apply the filter.
    target: scalar or array_like
        Set of classes labels to focus the filter on.
    indices:
        Nearest neihbors indices for each entry in classes.

    Returns
    -------
    c_maj: array_like
        1D set of filtered classes labels.

    """

    arr = np.vstack((arr_1, arr_2))

    class_1 = np.full(arr_1.shape[0], 1, dtype=np.int)
    class_2 = np.full(arr_2.shape[0], 2, dtype=np.int)
    classes = np.hstack((class_1, class_2)).T

    indices = set_nbrs_knn(arr, arr, knn, return_dist=False)

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


def class_filter_rad(arr_1, arr_2, rad, target):

    """
    Function to apply a majority filter on a set of classes, while focusing on
    a class.

    Parameters
    ----------
    classes: array_like
        1D set of classes labels to apply the filter.
    target: scalar or array_like
        Set of classes labels to focus the filter on.
    indices:
        Nearest neihbors indices for each entry in classes.

    Returns
    -------
    c_maj: array_like
        1D set of filtered classes labels.

    """

    arr = np.vstack((arr_1, arr_2))

    class_1 = np.full(arr_1.shape[0], 1, dtype=np.int)
    class_2 = np.full(arr_2.shape[0], 2, dtype=np.int)
    classes = np.hstack((class_1, class_2)).T

    indices = set_nbrs_rad(arr, arr, rad, return_dist=False)

    # Allocating output variable.
    c_maj = classes.copy()

    # Checking for the target class.
    target_idx = np.where(classes == np.array(target).any())[0]

    # Looping over the target points to filter.
    for i in target_idx:

        class_ = classes[indices[i]]
        # Counting the number of occurrences of each value in the ith instance
        # of class_.
        count = np.bincount(class_)
        c_maj[i] = count.argmax()

        # Appending the majority class into the output variable.
        c_maj[i] = count.argmax()

    return arr[c_maj == 1], arr[c_maj == 2]


def array_majority_rad(arr_1, arr_2, rad):

    """
    Function to apply majority filter on two arrays.

    Parameters
    ----------
    arr_1: array
        n-dimensional array of points to filter.
    arr_2: array
        n-dimensional array of points to filter.
    knn: int
        Number neighbors to select the subset of points to apply the
        majority criteria.

    Returns
    -------
    c_maj_1: array
        Filtered n-dimensional array of the same class as the input 'arr_1'.
    c_maj_2: array
        Filtered n-dimensional array of the same class as the input 'arr_2'.

    """
    # Stacking the arrays from both classes to generate a combined array.
    arr = np.vstack((arr_1, arr_2))

    # Generating the indices for the local subsets of points around all points
    # in the combined array.
    indices = set_nbrs_rad(arr, arr, rad, return_dist=False)

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


def array_majority(arr_1, arr_2, knn):

    """
    Function to apply majority filter on two arrays.

    Parameters
    ----------
    arr_1: array
        n-dimensional array of points to filter.
    arr_2: array
        n-dimensional array of points to filter.
    knn: int
        Number neighbors to select the subset of points to apply the
        majority criteria.

    Returns
    -------
    c_maj_1: array
        Filtered n-dimensional array of the same class as the input 'arr_1'.
    c_maj_2: array
        Filtered n-dimensional array of the same class as the input 'arr_2'.

    """
    # Stacking the arrays from both classes to generate a combined array.
    arr = np.vstack((arr_1, arr_2))

    # Generating the indices for the local subsets of points around all points
    # in the combined array.
    indices = set_nbrs_knn(arr, arr, knn, return_dist=False)

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
