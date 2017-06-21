# -*- coding: utf-8 -*-
"""
This is the main module in the tlseparation package. This module
manages all the processing in the separation algorithm.

@author: Matheus Boni Vicari (matheus.boni.vicari@gmail.com)
"""

import numpy as np
import pandas as pd
from classification.point_features import geodescriptors
from classification.gmm import classify
from utility.filtering import continuity_filter
from utility.filtering import majority
from utility.filtering import class_filter
from utility.filtering import array_majority
from utility.filtering import dist_majority_knn
from utility.point_compare import get_diff
from utility.continuous_clustering import path_clustering
from utility.knnsearch import set_nbrs_knn
from utility.knnsearch import subset_nbrs
from time import time


def main(arr, knn, class_file, slice_length=0.03, cluster_threshold=0.1,
         knn_downsample=1, freq_threshold=0.8, cont_filter=False):

    """
    Main function of the tlseparation package. This function manages the
    separation steps from an input array to the output of two separated
    arrays, wood and leaf.

    Parameters
    ----------
    arr: array
        Three-dimensional point cloud of a single tree to perform the wood-leaf
        separation. This should be a n-dimensional array (m x n) containing a
        set of coordinates (n) over a set of points (m).
    knn: int
        Number of nearest neighbors to search in order to constitue the local
        subset of points around each point in 'arr' for the first separation
        step.
    class_file: str
        Path to the classes reference values file. This file will be loaded
        and its reference values are used to select wood and leaf classes.
    slice_length: float
        Length of the slices of the data in 'arr'.
    cluster_threshold: float
        Distance threshold to be used as constraint in the slice clustering
        step.
    knn_downsample: float
        Downsample factor (0, 1) for the knn parameter. If less than 1, a
        sample of size (knn * knn_downsample) will be selected from the nearest
        neighbors indices. This option aims to maintain the spatial
        representation of the local subsets of points, but reducing overhead
        in memory and processing time.
    freq_threshold: float
        Threshold to separate the initial set of trunk/larger branches based
        on the log frequency of paths that passes through each node in the
        shortest path calculation. Valid values are between 0 and 1.
    cont_filter: bool
        Option to set if the continuity filter should run or not.

    Returns
    -------
    wood: array
        Three-dimensional array of points classified as wood.
    leaf: array
        Three-dimensional array of points classified as leaf.
    p: list
        Set of parameter used to perform the separation.

    """
    
    knn_lst = [knn * 0.7, knn * 0.85, knn, knn * 1.15, knn * 1.3]
    knn_lst = np.array(knn_lst).astype(int)
    vote_threshold = int(len(knn_lst))

    # Setting initial processing time.
    proc0 = time()

    # To avoid a common error, set knn variable as int.
    knn = int(knn)

    # Removing duplicates from input data.
    print('\nRemoving duplicates from input point cloud.')
    size0 = arr.shape[0]
    arr = remove_duplicates(arr)
    print('%s points removed.\n' % (size0 - arr.shape[0]))

    # Initial step of the automatic separation to make sure the trunk
    # and larger branches are detected as wood.
    print('Performing path clustering to detect trunk and main branches.')
    t0 = time()
    wood_s, leaf_s = path_clustering(arr, 20, slice_length, cluster_threshold,
                                     freq_threshold)
    print('Path clustering completed, %s seconds elapsed.' % (time() - t0))
    print('%s points detected.\n' % (size0 - wood_s.shape[0]))

    # Try to separate using wlseparate_abs, which select wood classes
    # based on an absolute value as threshold in the parameter space.
    print('Performing wood-leaf separation with absolute threshold (ABS).')
    t0 = time()
    try:
        wood_1, leaf_1 = wlseparate_abs(arr, 50, n_classes=3)
        wood = np.vstack((wood_s, wood_1))
    except:
        wood = wood_s
    print('Wood-leaf ABS separation completed, %s seconds elapsed.' %
          (time() - t0))
    print('%s points detected.\n' % (wood.shape[0] - wood_s.shape[0]))

    # Try to separate using wlseparate_ref, which uses reference values
    # to select the most likely classes to be wood or leaf.
    print('Performing wood-leaf separation with class reference (REF).')
    t0 = time()
    size0 = wood.shape[0]
    try:
        wood_2, leaf_2 = wlseparate_ref(arr, knn, class_file)
        wood = np.vstack((wood, wood_2))
    except:
        pass
    print('Wood-leaf REF separation completed, %s seconds elapsed.' %
          (time() - t0))
    print('%s points detected.\n' % (abs(size0 - wood.shape[0])))
    
    # Try to separate using wlseparate_ref_voting, which uses reference values
    # and a voting scheme to select the most likely classes to be wood or leaf.
    print('Performing wood-leaf separation with class reference voting scheme\
 (VOTE-REF).')
    t0 = time()
    size0 = wood.shape[0]
    try:
        wood_3, leaf_3 = wlseparate_ref_voting(arr, knn_lst, class_file,
                                               vote_threshold, n_classes=4)
        wood_3, leaf_3 = array_majority(wood_3, leaf_3, knn)
        wood = np.vstack((wood, wood_3))
    except:
        pass
    print('Wood-leaf REF separation completed, %s seconds elapsed.' %
          (time() - t0))
    print('%s points detected.\n' % (abs(size0 - wood.shape[0])))    
    
    
    # Removing duplicates from wood points and selecting generating leaf
    # points variabl from the difference set of wood and input array.
    wood = remove_duplicates(wood)
    leaf = get_diff(wood, arr)

    # Applying majority filter to wood and leaf points in order to remove
    # isolated points.
    print('Starting majority filter.')
    t0 = time()
    wood, leaf = array_majority(wood, leaf, knn)
    print('Majority filter completed, %s seconds elapsed.\n' % (time() - t0))
    
    # If set True, runs a continuity filter over the wood and leaf point
    # clouds.
    if cont_filter is True:
        print('Starting continuity filter.')
        t0 = time()
        radius = detect_nn_dist(wood, 2)
        size0 = wood.shape[0]
        wood, leaf = continuity_filter(wood, leaf, rad=radius)

        # Removing duplicates from wood and leaf.
        wood = remove_duplicates(wood)
        leaf = remove_duplicates(leaf)
        print('Continuity filter completed, %s seconds elapsed.' %
              (time() - t0))
        print('%s new points detected as wood.\n' %
              (abs(size0 - wood.shape[0])))
       
    # Running distance majority filter. This filter aims to make the cloud
    # more uniform by evaluating the sum of distances of each class in the
    # neighboring points.
    print('Starting weighted majority filter.')
    t0 = time()
    wood, leaf = dist_majority_knn(wood, leaf, knn)
    print('Wighted majority filter completed, %s seconds elapsed.\n' %
          (time() - t0))
    
    # Setting the parameters list to output. Used to control the settings
    # of the processing.
    p = [knn, slice_length, cluster_threshold, knn_downsample, freq_threshold]
    
    print('Processing finished, %s seconds elapsed.\n' % (time() - proc0))
    print('Wood: %s points.' % wood.shape[0])
    print('Leaf: %s points.\n' % leaf.shape[0])

    return wood, leaf, p
  
      
def wlseparate_ref_voting(arr, knn_lst, class_file, threshold, n_classes):
    
    vt = []
    
    d_base, idx_base = set_nbrs_knn(arr, arr, np.max(knn_lst),
                                    return_dist=True)
                                    
    for k in knn_lst:
        dx_1, idx_1 = subset_nbrs(d_base, idx_base, k)
        
        # Calculating the geometric descriptors.
        gd_1 = geodescriptors(arr, idx_1)
    
        # Normalizing geometric descriptors
        gd_1 = ((gd_1 - np.min(gd_1, axis=0)) /
                (np.max(gd_1, axis=0) - np.min(gd_1, axis=0)))
    
        # Classifying the points based on the geometric descriptors.
        classes_1, cm_1 = classify(gd_1, n_classes)
    
        # Selecting which classes represent wood and leaf. Wood classes are masked
        # as True and leaf classes as False.
        class_table = pd.read_csv(class_file)
        class_ref = np.asarray(class_table.ix[:, 1:]).astype(float)
        new_classes = class_select(classes_1, cm_1, class_ref)
        
        # Appending results to vt temporary list.
        vt.append((new_classes == 1) | (new_classes == 2))
    
    
    idf = np.array(vt[0], ndmin=2).T
    for i in vt[1:]:
        idf = np.hstack((idf, np.array(i, ndmin=2).T))
    votes = np.sum(idf, axis=1)
#        
    mask = votes >= threshold
    
    return arr[mask], arr[~mask]


def wlseparate_ref(arr, knn, class_file, knn_downsample=1,
                   n_classes=3):

    """
    Primary wood-leaf separation code.

    Parameters
    ----------
    arr: array
        Three-dimensional point cloud of a single tree to perform the wood-leaf
        separation. This should be a n-dimensional array (m x n) containing a
        set of coordinates (n) over a set of points (m).
    knn: int
        Number of nearest neighbors to search to constitue the local subset of
        points around each point in 'arr'.
    knn_downsample: float
        Downsample factor (0, 1) for the knn parameter. If less than 1, a
        sample of size (knn * knn_downsample) will be selected from the nearest
        neighbors indices. This option aims to maintain the spatial
        representation of the local subsets of points, but reducing overhead
        in memory and processing time.
    n_classes: int
        Number of classes to use in the Gaussian Mixture Classification.

    Returns
    -------
    wood: array
        Three-dimensional array of points classified as wood.
    leaf: array
        Three-dimensional array of points classified as leaf.

    """

    # Generating the indices array of the 'k' nearest neighbors (knn) for all
    # points in arr.
    idx_1 = set_nbrs_knn(arr, arr, knn, return_dist=False)

    if knn_downsample < 1:
        n_samples = np.int(idx_1.shape[1] * knn_downsample)
        idx_f = np.zeros([idx_1.shape[0], n_samples + 1])
        idx_f[:, 0] = idx_1[:, 0]
        for i in range(idx_f.shape[0]):
            idx_f[i, 1:] = np.random.choice(idx_1[i, 1:], n_samples,
                                            replace=False)
        idx_1 = idx_f.astype(int)

    # Calculating the geometric descriptors.
    gd_1 = geodescriptors(arr, idx_1)

    # Normalizing geometric descriptors
    gd_1 = ((gd_1 - np.min(gd_1, axis=0)) /
            (np.max(gd_1, axis=0) - np.min(gd_1, axis=0)))

    # Classifying the points based on the geometric descriptors.
    classes_1, cm_1 = classify(gd_1, n_classes)

    # Selecting which classes represent wood and leaf. Wood classes are masked
    # as True and leaf classes as False.
    class_table = pd.read_csv(class_file)
    class_ref = np.asarray(class_table.ix[:, 1:]).astype(float)
    new_classes = class_select(classes_1, cm_1, class_ref)

    mask_1 = (new_classes == 1) | (new_classes == 2)

    # Returning the wood and leaf points based on the class selection mask.
    # mask represent wood points, (~) not mask represent leaf points.
    return arr[mask_1, :], arr[~mask_1, :]


def wlseparate_abs(arr, knn, knn_downsample=1, n_classes=2):

    """
    Primary wood-leaf separation code.

    Parameters
    ----------
    arr: array
        Three-dimensional point cloud of a single tree to perform the wood-leaf
        separation. This should be a n-dimensional array (m x n) containing a
        set of coordinates (n) over a set of points (m).
    knn: int
        Number of nearest neighbors to search to constitue the local subset of
        points around each point in 'arr'.
    knn_downsample: float
        Downsample factor (0, 1) for the knn parameter. If less than 1, a
        sample of size (knn * knn_downsample) will be selected from the nearest
        neighbors indices. This option aims to maintain the spatial
        representation of the local subsets of points, but reducing overhead
        in memory and processing time.
    n_classes: int
        Number of classes to use in the Gaussian Mixture Classification.

    Returns
    -------
    wood: array
        Three-dimensional array of points classified as wood.
    leaf: array
        Three-dimensional array of points classified as leaf.

    """

    # Generating the indices array of the 'k' nearest neighbors (knn) for all
    # points in arr.
    idx_1 = set_nbrs_knn(arr, arr, knn, return_dist=False)

    if knn_downsample < 1:
        n_samples = np.int(idx_1.shape[1] * knn_downsample)
        idx_f = np.zeros([idx_1.shape[0], n_samples + 1])
        idx_f[:, 0] = idx_1[:, 0]
        for i in range(idx_f.shape[0]):
            idx_f[i, 1:] = np.random.choice(idx_1[i, 1:], n_samples,
                                            replace=False)
        idx_1 = idx_f.astype(int)

    # Calculating the geometric descriptors.
    gd_1 = geodescriptors(arr, idx_1)

    # Classifying the points based on the geometric descriptors.
    classes_1, cm_1 = classify(gd_1, n_classes)

    # Selecting which classes represent wood and leaf. Wood classes are masked
    # as True and leaf classes as False.
    mask_1 = class_select_abs(classes_1, cm_1, idx_1, filt='all')

    # Returning the wood and leaf points based on the class selection mask.
    # mask represent wood points, (~) not mask represent leaf points.
    return arr[mask_1, :], arr[~mask_1, :]


def class_select(classes, cm, classes_ref):

    """
    Function to select from the classification results which classes
    are wood and which are leaf.

    Parameters
    ----------
    classes: list
        List of classes labels for each observation from the input variables.
    cm: array
        N-dimensional array (c x n) of each class (c) parameter space mean
        valuess (n).
    nbrs_idx: array_like
        Nearest Neighbors indices relative to every point of the array
        that originated the classes labels.
    filt: string
        Option to filter or not the classes labels.

    Returns
    -------
    mask: list
        List of booleans where True represents wood points and False
        represents leaf points.

    """

    class_ids = np.zeros([cm.shape[0]])
    for c in range(cm.shape[0]):
        mindist = np.inf
        for i in range(classes_ref.shape[0]):
            d = np.linalg.norm(cm[c] - classes_ref[i])
            if d < mindist:
                class_ids[c] = i
                mindist = d

    new_classes = np.zeros([classes.shape[0]])
    for i in range(new_classes.shape[0]):
        new_classes[i] = class_ids[classes[i]]

    return new_classes


def class_select_abs(classes, cm, nbrs_idx, filt=None):

    """
    Function to select from the classification results which classes
    are wood and which are leaf.

    Parameters
    ----------
    classes: list
        List of classes labels for each observation from the input variables.
    cm: array
        N-dimensional array (c x n) of each class (c) parameter space mean
        valuess (n).
    nbrs_idx: array_like
        Nearest Neighbors indices relative to every point of the array
        that originated the classes labels.
    filt: string
        Option to filter or not the classes labels.

    Returns
    -------
    mask: list
        List of booleans where True represents wood points and False
        represents leaf points.

    """

    # Calculating the ratio of first 3 components of the classes means (cm).
    # These components are the basic geometric descriptors.
    if np.max(np.sum(cm, axis=1)) >= 0.5:

        class_id = np.argmax(cm[:, 5])

        if filt == 'class':
            classes = class_filter(classes, class_id, nbrs_idx)
        elif filt == 'all':
            classes = majority(classes, nbrs_idx)

        # Masking classes based on the criterias set above. Mask will present
        # True for wood points and False for leaf points.
        mask = classes == class_id

    else:
        mask = []

    return mask


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
