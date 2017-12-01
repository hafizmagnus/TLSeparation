# -*- coding: utf-8 -*-
"""
Created on Sun Sep 17 19:21:03 2017

@author: Matheus
"""

import numpy as np
from knnsearch import set_nbrs_knn


def detect_nn_dist(arr, knn):

    """
    Function to calculate the optimum distance among neighboring points.

    Parameters
    ----------
    arr: array
        N-dimensional array (m x n) containing a set of parameters (n) over a
        set of observations (m).
    knn: int
        Number of nearest neighbors to search to constitue the local subset of
        points around each point in 'arr'.

    Returns
    -------
    dist: float
        Optimal distance among neighboring points.

    """

    dist, indices = set_nbrs_knn(arr, arr, knn)

    return np.mean(dist[:, 1:]) + (np.std(dist[:, 1:]))
