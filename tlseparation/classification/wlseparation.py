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
from pandas import read_csv
from sklearn.neighbors import NearestNeighbors
import sys

sys.path.append('..')

from utility.knnsearch import set_nbrs_knn, subset_nbrs
from classification.point_features import geodescriptors
from classification.gmm import (classify, class_select_abs,
                                class_select_ref)


def fill_class(arr1, arr2, noclass, k):

    """
    Assigns noclass entries to either arr1 or arr2, depending on
    neighborhood majority analisys.

    Args:
        arr1 (array): Point coordinates for entries of the first class.
        arr2 (array): Point coordinates for entries of the second class.
        noclass (array): Point coordinates for noclass entries.
        k (int): Number of neighbors to use in the neighborhood majority
            analysis.

    Returns:
        arr1 (array): Point coordinates for entries of the first class.
        arr2 (array): Point coordinates for entries of the second class.

    """

    # Stacking arr1 and arr2. This will be fitted in the NearestNeighbors
    # search in order to define local majority and assign classes to
    # noclass.
    arr = np.vstack((arr1, arr2))

    # Generating classes labels with the same shapes as arr1, arr2 and,
    # after stacking, arr.
    class_1 = np.full(arr1.shape[0], 1, dtype=np.int)
    class_2 = np.full(arr2.shape[0], 2, dtype=np.int)
    classes = np.hstack((class_1, class_2)).T

    # Performin NearestNeighbors search to detect local sets of points.
    nbrs = NearestNeighbors(leaf_size=25, n_jobs=-1).fit(arr)
    indices = nbrs.kneighbors(noclass, n_neighbors=k, return_distance=False)

    # Allocating output variable.
    new_class = np.zeros(noclass.shape[0])

    # Selecting subset of classes based on the neighborhood expressed by
    # indices.
    class_ = classes[indices]

    # Looping over all points in indices.
    for i in range(len(indices)):

        # Counting the number of occurrences of each value in the ith instance
        # of class_.
        unique, count = np.unique(class_[i, :], return_counts=True)
        # Appending the majority class into the output variable.
        new_class[i] = unique[np.argmax(count)]

    # Stacking new points to arr1 and arr2.
    arr1 = np.vstack((arr1, noclass[new_class == 1]))
    arr2 = np.vstack((arr2, noclass[new_class == 2]))

    # Making sure all points were processed and assigned a class.
    assert ((arr1.shape[0] + arr2.shape[0]) ==
            (arr.shape[0] + noclass.shape[0]))

    return arr1, arr2


def wlseparate_ref_voting(arr, knn_lst, class_file, n_classes=3,
                          prob_threshold=0.95):

    """
    Classifies a point cloud (arr) into two main classes, wood and leaf.
    Altough this function does not output a noclass category, it still
    filters out results based on classification confidence interval in the
    voting process (if lower than prob_threshold, then voting is not used
    for current point and knn value).

    The final class selection is based a voting scheme applied to a similar
    approach of wlseparate_ref. In this case, the function iterates over a
    series of knn values and apply the reference distance criteria to select
    wood and leaf classes.

    Each knn class result is accumulated in a list and in the end a voting
    is applied. For each point, if the number of times it was classified as
    wood is larger than threhsold, the final class is set to wood. Otherwise
    it is set as leaf.

    Class selection will mask points according to their class mean distance
    to reference classes. The closes reference class gets assignes to each
    intermediate class.

    Args:
        arr (array): Three-dimensional point cloud of a single tree to perform
            the wood-leaf separation. This should be a n-dimensional array
            (m x n) containing a set of coordinates (n) over a set of points
            (m).
        knn_lst (list): List of knn values to use in the search to constitue
            local subsets of points around each point in 'arr'.
        class_file (str): Path to reference classes file.
        n_classes (int): Number of classes to use in the Gaussian Mixture
            Classification.
        prob_threshold (float): Probability threshold to select if points
            are classified within a confidence interval or not.

    Returns:
        class_dict (dict): Dictionary containing all classses labeled
            according to class names in class_file.

    """

    # Initializing voting accumulator list.
    vt = np.full([arr.shape[0], len(knn_lst)], -1, dtype=int)

    # Generating a base set of indices and distances around each point.
    # This step uses the largest value in knn_lst to make further searches,
    # with smaller values of knn, more efficient.
    d_base, idx_base = set_nbrs_knn(arr, arr, np.max(knn_lst),
                                    return_dist=True)

    # Reading in class reference values from file.
    class_table = read_csv(class_file)
    class_ref = np.asarray(class_table.iloc[:, 1:]).astype(float)

    # Looping over values of knn in knn_lst.
    for i, k in enumerate(knn_lst):
        # Subseting indices and distances based on initial knn search and
        # current knn value (k).
        dx_1, idx_1 = subset_nbrs(d_base, idx_base, k)

        # Calculating the geometric descriptors.
        gd_1 = geodescriptors(arr, idx_1)

        # Classifying the points based on the geometric descriptors.
        classes_1, cm_1, proba_1 = classify(gd_1, n_classes)
        cm_1 = ((cm_1 - np.min(cm_1, axis=0)) /
                (np.max(cm_1, axis=0) - np.min(cm_1, axis=0)))
        # Masking classes according to their predicted posterior probability of
        # belonging on each class. If probability is smaller than a given
        # threshold, point will be masked as not classified.
        prob_mask = np.max(proba_1, axis=1) >= prob_threshold

        # Selecting which classes represent classes from classes reference
        # file.
        new_classes = class_select_ref(classes_1, cm_1, class_ref)

        # Appending results to vt temporary list.
        noclass_prob_ids = np.where(~prob_mask)[0]
        vt[:, i] = new_classes.astype(int)
        vt[noclass_prob_ids, i] = -1

    # Performing the voting scheme (majority selection) for each point.
    final_class = np.full([arr.shape[0]], -1, dtype=int)
    for i, v in enumerate(vt):
        unique, count = np.unique(v, return_counts=True)
        final_class[i] = unique[np.argmax(count)]

    # Generating new classification probability mask.
    prob_mask = final_class != -1

    # Creating output class dictionary. Class names are the same as from
    # class_file.
    class_dict = {}
    for c in np.unique(final_class).astype(int):
        class_data = arr[prob_mask][final_class[prob_mask] == c]
        class_dict[class_table.iloc[c, :]['class']] = class_data
    class_dict['noclass'] = arr[~prob_mask]

    return class_dict


def wlseparate_ref(arr, knn, class_file, knn_downsample=1,
                   n_classes=3, prob_threshold=0.95):

    """
    Classifies a point cloud (arr) into three main classes, wood, leaf and
    noclass.

    The final class selection is based on distance in the parameter space
    of every intermediate class (n_classes) and reference classes from a
    file (class_file).

    Points will be only classified as wood or leaf if their classification
    probability is higher than prob_threshold. Otherwise, points are
    assigned to noclass.

    Class selection will mask points according to their class mean distance
    to reference classes. The closes reference class gets assignes to each
    intermediate class.

    Args:
        arr (array): Three-dimensional point cloud of a single tree to perform
            the wood-leaf separation. This should be a n-dimensional array
            (m x n) containing a set of coordinates (n) over a set of points
            (m).
        knn (int): Number of nearest neighbors to search to constitue the
            local subset of points around each point in 'arr'.
        class_file (str): Path to reference classes file.
        knn_downsample (float): Downsample factor (0, 1) for the knn
            parameter. If less than 1, a sample of size (knn * knn_downsample)
            will be selected from the nearest neighbors indices. This option
            aims to maintain the spatial representation of the local subsets
            of points, but reducing overhead in memory and processing time.
        n_classes (int): Number of classes to use in the Gaussian Mixture
            Classification.
        prob_threshold (float): Probability threshold to select if points
            are classified within a confidence interval or not.

    Returns:
        class_dict (dict): Dictionary containing all classses labeled
            according to class names in class_file.

    """

    # Generating the indices array of the 'k' nearest neighbors (knn) for all
    # points in arr.
    idx_1 = set_nbrs_knn(arr, arr, knn, return_dist=False)

    # If downsample fraction value is set to lower than 1. Apply downsampling
    # on knn indices.
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
    classes_1, cm_1, proba_1 = classify(gd_1, n_classes)
    cm_1 = ((cm_1 - np.min(cm_1, axis=0)) /
            (np.max(cm_1, axis=0) - np.min(cm_1, axis=0)))
    # Masking classes according to their predicted posterior probability of
    # belonging on each class. If probability is smaller than a given
    # threshold, point will be masked as not classified.
    prob_mask = np.max(proba_1, axis=1) >= prob_threshold

    # Selecting which classes represent wood and leaf. Wood classes are masked
    # as True and leaf classes as False.
    class_table = read_csv(class_file)
    class_ref = np.asarray(class_table.ix[:, 1:]).astype(float)
    new_classes = class_select_ref(classes_1[prob_mask], cm_1, class_ref)

    # Creating output class dictionary. Class names are the same as from
    # class_file.
    class_dict = {}
    for c in np.unique(new_classes).astype(int):
        class_data = arr[prob_mask][new_classes == c]
        class_dict[class_table.iloc[c, :]['class']] = class_data
    class_dict['noclass'] = arr[~prob_mask]

    return class_dict


def wlseparate_abs(arr, knn, knn_downsample=1, n_classes=2,
                   prob_threshold=0.95):

    """
    Classifies a point cloud (arr) into three main classes, wood, leaf and
    noclass.

    The final class selection is based on the absolute value of the last
    geometric feature (see point_features module).
    Points will be only classified as wood or leaf if their classification
    probability is higher than prob_threshold. Otherwise, points are
    assigned to noclass.

    Class selection will mask points with feature value larger than a given
    threshold as wood and the remaining points as leaf.

    Args:
        arr (array): Three-dimensional point cloud of a single tree to perform
            the wood-leaf separation. This should be a n-dimensional array
            (m x n) containing a set of coordinates (n) over a set of points
            (m).
        knn (int): Number of nearest neighbors to search to constitue the
            local subset of points around each point in 'arr'.
        knn_downsample (float): Downsample factor (0, 1) for the knn
            parameter. If less than 1, a sample of size (knn * knn_downsample)
            will be selected from the nearest neighbors indices. This option
            aims to maintain the spatial representation of the local subsets
            of points, but reducing overhead in memory and processing time.
        n_classes (int): Number of classes to use in the Gaussian Mixture
            Classification.
        prob_threshold (float): Probability threshold to select if points
            are classified within a confidence interval or not.

    Returns:
        class_dict (dict): Dictionary containing wood, leaf and noclass
            classes.

    """

    # Generating the indices array of the 'k' nearest neighbors (knn) for all
    # points in arr.
    idx_1 = set_nbrs_knn(arr, arr, knn, return_dist=False)

    # If downsample fraction value is set to lower than 1. Apply downsampling
    # on knn indices.
    if knn_downsample < 1:
        n_samples = np.int(idx_1.shape[1] * knn_downsample)
        idx_f = np.zeros([idx_1.shape[0], n_samples + 1])
        idx_f[:, 0] = idx_1[:, 0]
        for i in range(idx_f.shape[0]):
            idx_f[i, 1:] = np.random.choice(idx_1[i, 1:], n_samples,
                                            replace=False)
        idx_1 = idx_f.astype(int)

    # Calculating geometric descriptors.
    gd_1 = geodescriptors(arr, idx_1)

    # Classifying the points based on the geometric descriptors.
    classes_1, cm_1, proba_1 = classify(gd_1, n_classes)
    # Masking classes according to their predicted posterior probability of
    # belonging on each class. If probability is smaller than a given
    # threshold, point will be masked as not classified.
    prob_mask = np.max(proba_1, axis=1) >= prob_threshold

    # Selecting which classes represent wood and leaf. Wood classes are masked
    # as True and leaf classes as False.
    mask_1 = class_select_abs(classes_1[prob_mask], cm_1,
                              idx_1[prob_mask])

    # Creating output class dictionary. Class names are the same as from
    # class_file.
    # mask represent wood points, (~) not mask represent leaf points.
    class_dict = {}
    class_dict['wood'] = arr[prob_mask][mask_1, :]
    class_dict['leaf'] = arr[prob_mask][~mask_1, :]
    class_dict['noclass'] = arr[~prob_mask]

    return class_dict
