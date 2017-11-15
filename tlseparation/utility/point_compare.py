# -*- coding: utf-8 -*-
"""
Created on Wed Aug 24 08:38:19 2016

@author: mathe
"""

import numpy as np
import pandas as pd


def get_diff(arr1, arr2):

    """
    Function to perform the intersection of two arrays, returning the
    entries not intersected between arr1 and arr2.

    Parameters
    ----------
    arr1: array
        N-dimensional array of points to intersect.
    arr2: array
        N-dimensional array of points to intersect.

    Returns
    -------
    arr: array
        Difference array between 'arr1' and 'arr2'.

    """

    # Asserting that both arrays have the same number of columns.
    assert arr1.shape[1] == arr2.shape[1]

    # Stacking both arrays.
    arr3 = np.vstack((arr1, arr2))

    # Creating a pandas.DataFrame from the stacked array.
    df = pd.DataFrame(arr3)

    # Removing duplicate points and keeping only points that have only a
    # single occurrence in the stacked array.
    diff = df.drop_duplicates(keep=False)

    return np.asarray(diff)


def remove_duplicates(arr, return_ids=False):

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
    df = pd.DataFrame({'x': arr[:, 0],
                       'y': arr[:, 1], 'z': arr[:, 2]})

    # Using the drop_duplicates function to remove the duplicate points from
    # df.
    unique = df.drop_duplicates(['x', 'y', 'z'])

    return np.asarray(unique).astype(float)