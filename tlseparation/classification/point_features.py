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


def geodescriptors(arr, nbr_idx):

    """
    Function to calculate the geometric descriptors: salient features and
    tensor features.

    Args:
        arr (array): Three-dimensional (m x n) array of a point cloud, where
            the coordinates are represented in the columns (n) and the points
            are represented in the rows (m).
        nbr_idx (array): N-dimensional array of indices from a nearest
            neighbors search of the point cloud in 'arr', where the rows (m)
            represents the points in 'arr' and the columns represents the
            indices of the nearest neighbors from 'arr'.

    Returns:
        features (array): N-dimensional array (m x 6) of the calculated
            geometric descriptors. Where the rows (m) represent the points
            from 'arr' and the columns represents the features.

    """

    # Making sure nbr_idx has the correct data type.
    nbr_idx = nbr_idx.astype(int)

    # Calculating the eigenvalues.
    s = eigen(arr[nbr_idx])

    # Calculating the ratio of the eigenvalues.
    ratio = (s.T / np.sum(s, axis=1)).T

    # Calculating the salient features and tensor features from the
    # eigenvalues ratio.
    features = calc_features(ratio)

    # Replacing the 'nan' values for 0.
    features[np.isnan(features)] = 0

    return features


def eigen(arr_stack):

    """
    Function to calculate the eigenvalues of a stack of arrays.

    Args:
        arr_stack (array): N-dimensional array (l x m x n) containing a
            stack of data, where the rows (m) represents the points
            coordinates, the columns (n) represents the axis coordinates and
            the layer (l) represents the stacks of points.

    Returns:
        evals (array): N-dimensional array (l x n) of eigenvalues calculated
            from 'arr_stack'. The rows (l) represents the stack layers of
            points in 'arr_stack' and the columns (n) represent the parameters
            in 'arr_stack'.

    """

    # Calculating the covariance of the stack of arrays.
    cov = vectorized_app(arr_stack)

    # Calculating the eigenvalues using Singular Value Decomposition (svd).
    evals = np.linalg.svd(cov, compute_uv=False)

    return evals


def calc_features(e):

    """
    Calculates the geometric features using a set of eigenvalues, based on Ma
    et al. (2015) and Wang et al. (2015).

    Args:
        e (array): N-dimensional array (m x 3) containing sets of 3
            eigenvalues per row (m).

    Returns:
        features (array): N-dimensional array (m x 6) containing the
            calculated geometric features from 'e'.

    Reference:
    ..  [1] Ma et al., 2015. Improved Salient Feature-Based Approach for
            Automatically Separating Photosynthetic and Nonphotosynthetic
            Components Within Terrestrial Lidar Point Cloud Data of Forest
            Canopies.
    ..  [2] Wang et al., 2015. A Multiscale and Hierarchical Feature Extraction
            Method for Terrestrial Laser Scanning Point Cloud Classification.

    """

    # Calculating salient features.
    e1 = e[:, 2]
    e2 = e[:, 0] - e[:, 1]
    e3 = e[:, 1] - e[:, 2]

    # Calculating tensor features.
    t1 = (e[:, 1] - e[:, 2]) / e[:, 0]
    t2 = ((e[:, 0] * np.log(e[:, 0])) + (e[:, 1] * np.log(e[:, 1])) +
          (e[:, 2] * np.log(e[:, 2])))
    t3 = (e[:, 0] - e[:, 1]) / e[:, 0]

    return np.vstack(([e1, e2, e3, t1, t2, t3])).T


def vectorized_app(arr_stack):

    """
    Function to calculate the covariance of a stack of arrays. This function
    uses einstein summation to make the covariance calculation more efficient.
    Based on a reply from the user Divakar at stackoverflow.

    Args:
        arr_stack (array): N-dimensional array (l x m x n) containing a stack
            of data, where the rows (m) represents the points coordinates, the
            columns (n) represents the axis coordinates and the layer (l)
            represents the stacks of points.

    Returns:
        cov (array): N-dimensional array (l x n x n) of covariance values
            calculated from 'arr_stack'. Each layer (l) contains a (n x n)
            covariance matrix calculated from the layers (l) in 'arr_stack'.

    Reference:
    ..  [1] Divakar, 2016. http://stackoverflow.com/questions/35756952/\
quickly-compute-eigenvectors-for-each-element-of-an-array-in-\
python.

    """

    # Centralizing the data around the mean.
    diffs = arr_stack - arr_stack.mean(1, keepdims=True)

    # Using the einstein summation of the centered data in regard to the array
    # stack shape to return the covariance of each array in the stack.
    return np.einsum('ijk,ijl->ikl', diffs, diffs)/arr_stack.shape[1]
