# -*- coding: utf-8 -*-
"""
Module to manage the calculation of the pointwise geometric features
for a 3D point cloud.

@author: Matheus Boni Vicari (matheus.boni.vicari@gmail.com)
"""

import numpy as np


def geodescriptors(arr, nbr_idx, norm=False):

    """
    Function to calculate the geometric descriptors: salient features and
    tensor features.

    Parameters
    ----------
    arr: array_like
        Three-dimensional (m x n) array of a point cloud, where the coordinates
        are represented in the columns (n) and the points are represented in
        the rows (m).
    nbr_idx: array_like
        N-dimensional array of indices from a nearest neighbors search of the
        point cloud in 'arr', where the rows (m) represents the points in 'arr'
        and the columns represents the indices of the nearest neighbors from
        'arr'.
    norm: boolean
        Option to normalize the calculated salient features. Default is 'True'.
        Note that the tensor features are never normalized.

    Returns
    -------
    features: array_like
        N-dimensional array (m x 6) of the calculated geometric descriptors.
        Where the rows (m) represent the points from 'arr' and the columns
        represents the features.

    """

    # Calculating the salient features.
    features = fixed_neighbors(arr, nbr_idx)

    # Replacing the 'nan' values for 0.
    features[np.isnan(features)] = 0

    # Normalizing the salient features.
    if norm is True:
        features[:, :3] = (features[:, :3].T /
                           np.sum(features[:, :3], axis=1)).T

    return features


def fixed_neighbors(arr, nbr_idx):

    """
    Function to calculate the salient features using a fixed number of
    neighbors indices to select the subset of points around each point
    in the input 3D cloud.

    Parameters
    ----------
    arr: array_like
        Three-dimensional (m x n) array of a point cloud, where the coordinates
        are represented in the columns (n) and the points are represented in
        the rows (m).
    nbr_idx: array_like
        N-dimensional array of indices from a nearest neighbors search of the
        point cloud in 'arr', where the rows (m) represents the points in 'arr'
        and the columns represents the indices of the nearest neighbors from
        'arr'.

    Returns
    -------
    salient_features: array_like
        N-dimensional array (m x 3) of the calculated salient features.
        Where the rows (m) represent the points from 'arr' and the columns
        represents the features.
    tensor_features: array_like
        N-dimensional array (m x 3) of the calculated tensor features.
        Where the rows (m) represent the points from 'arr' and the columns
        represents the features.

    """

    # Calculating the eigenvalues.
    s = eigen(arr[nbr_idx])

    # Calculating the ratio of the eigenvalues.
    ratio = (s.T / np.sum(s, axis=1)).T

    # Calculating the salient features and tensor features from the
    # eigenvalues ratio.
    fa = calc_feature(ratio)
    ta = calc_tensor(ratio)

    return np.vstack((np.asarray(fa), np.asarray(ta))).T


def eigen(arr_stack):

    """
    Function to calculate the eigenvalues of a stack of arrays.

    Parameters
    ----------
    arr_stack: array_like
        N-dimensional array (l x m x n) containing a stack of data, where the
        rows (m) represents the points coordinates, the columns (n) represents
        the axis coordinates and the layer (l) represents the stacks of points.

    Returns
    -------
    evals: array_like
        N-dimensional array (l x n) of eigenvalues calculated from 'arr_stack'.
        The rows (l) represents the stack layers of points in 'arr_stack' and
        the columns (n) represent the parameters in 'arr_stack'.

    """

    # Calculating the covariance of the stack of arrays.
    cov = vectorized_app(arr_stack)

    # Calculating the eigenvalues using Singular Value Decomposition (svd).
    evals = np.linalg.svd(cov, compute_uv=False)

    return evals


def calc_feature(e):

    """
    Function to calculate the salient features using a set of eigenvalues,
    based on Ma et al., 2015.

    Parameters
    ----------
    e: array_like
        N-dimensional array (m x 3) containing sets of 3 eigenvalues per row
        (m).

    Returns
    -------
    features: array_like
        N-dimensional array (m x 3) containing the calculated salient features
        from 'e'.

    Reference
    ---------
    ..  [1] Ma et al., 2015. Improved Salient Feature-Based Approach for
            Automatically Separating Photosynthetic and Nonphotosynthetic
            Components Within Terrestrial Lidar Point Cloud Data of Forest
            Canopies.
    """
    return ([e[:, 2], e[:, 0] - e[:, 1], e[:, 1] - e[:, 2]])


def calc_tensor(e):

    """
    Function to calculate the tensor features using a set of eigenvalues,
    based on Wang et al., 2015.

    Parameters
    ----------
    e: array_like
        N-dimensional array (m x 3) containing sets of 3 eigenvalues per row
        (m).

    Returns
    -------
    features: array_like
        N-dimensional array (m x 3) containing the calculated tensor features
        from 'e'.

    Reference
    ---------
    ..  [1] Wang et al., 2015. A Multiscale and Hierarchical Feature Extraction
            Method for Terrestrial Laser Scanning Point Cloud Classification.

    """
    t1 = (e[:, 1] - e[:, 2]) / e[:, 0]
    t2 = ((e[:, 0] * np.log(e[:, 0])) + (e[:, 1] * np.log(e[:, 1])) +
          (e[:, 2] * np.log(e[:, 2])))
    t3 = (e[:, 0] - e[:, 1]) / e[:, 0]

    return t1, t2, t3


def vectorized_app(arr_stack):

    """
    Function to calculate the covariance of a stack of arrays. This function
    uses einstein summation to make the covariance calculation more efficient.
    Based on a reply from the user Divakar at stackoverflow.

    Parameters
    ----------
    arr_stack: array_like
        N-dimensional array (l x m x n) containing a stack of data, where the
        rows (m) represents the points coordinates, the columns (n) represents
        the axis coordinates and the layer (l) represents the stacks of points.

    Returns
    -------
    cov: array_like
        N-dimensional array (l x n x n) of covariance values calculated from
        'arr_stack'. Each layer (l) contains a (n x n) covariance matrix
        calculated from the layers (l) in 'arr_stack'.

    Reference
    ---------
    ..  [1] Divakar, 2016. http://stackoverflow.com/questions/35756952/\
quickly-compute-eigenvectors-for-each-element-of-an-array-in-\
python.
    """
    # Centralizing the data around the mean.
    diffs = arr_stack - arr_stack.mean(1, keepdims=True)

    # Using the eintein summation of the centered data in regard to the array
    # stack shape to return the covariance of each array in the stack.
    return np.einsum('ijk,ijl->ikl', diffs, diffs)/arr_stack.shape[1]
