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
__version__ = "1.1.5.1"
__maintainer__ = "Matheus Boni Vicari"
__email__ = "matheus.boni.vicari@gmail.com"
__status__ = "Development"

import numpy as np
from sklearn.mixture import GaussianMixture as GMM


def classify(variables, n_classes):

    """
    Function to perform the classification of a dataset using sklearn's
    Gaussian Mixture Models with Expectation Maximization.

    Args:
        variables (array): N-dimensional array (m x n) containing a set of
             parameters (n) over a set of observations (m).
        n_classes (int): Number of classes to assign the input variables.

    Returns:
        classes (list): List of classes labels for each observation from the
            input variables.
        means (array): N-dimensional array (c x n) of each class (c) parameter
            space means (n).
        probability (array): Probability of samples belonging to every class
             in the classification. Sum of sample-wise probability should be
             1.

    """

    gmm = GMM(n_components=n_classes)
    gmm.fit(variables)

    return gmm.predict(variables), gmm.means_, gmm.predict_proba(variables)


def class_select_ref(classes, cm, classes_ref):

    """
    Selects from the classification results which classes are wood and which
    are leaf.

    Args:
        classes (list): List of classes labels for each observation from the
            input variables.
        cm (array): N-dimensional array (c x n) of each class (c) parameter
            space mean valuess (n).
        classes_ref (array): Reference classes values.


    Returns:
        mask (array): List of booleans where True represents wood points and
            False represents leaf points.

    """

    # Initializing array of class ids.
    class_ids = np.zeros([cm.shape[0]])

    # Looping over each index in the classes means array.
    for c in range(cm.shape[0]):
        # Setting initial minimum distance value.
        mindist = np.inf
        # Looping over indices in classes reference values.
        for i in range(classes_ref.shape[0]):
            # Calculating distance of current class mean parameters and
            # current reference paramenters.
            d = np.linalg.norm(cm[c] - classes_ref[i])
            # Checking if current distance is smaller than previous distance
            # if so, assign current reference index to current class index.
            if d < mindist:
                class_ids[c] = i
                mindist = d

    # Assigning final classes values to new classes.
    new_classes = np.zeros([classes.shape[0]])
    for i in range(new_classes.shape[0]):
        new_classes[i] = class_ids[classes[i]]

    return new_classes


def class_select_abs(classes, cm, nbrs_idx, feature=5, threshold=0.5):

    """
    Select from GMM classification results which classes are wood and which
    are leaf based on a absolute value threshold from a single feature in
    the parameter space.

    Args:
        classes (list or array): Classes labels for each observation from the
            input variables.
        cm (array): N-dimensional array (c x n) of each class (c) parameter
            space mean valuess (n).
        nbrs_idx (array): Nearest Neighbors indices relative to every point
            of the array that originated the classes labels.
        feature (int): Column index of the feature to use as constraint.
        threshold (float): Threshold value to mask classes. All classes with
            means >= threshold are masked as true.

    Returns:
        mask (list): List of booleans where True represents wood points and
            False represents leaf points.

    """

    # Calculating the ratio of first 3 components of the classes means (cm).
    # These components are the basic geometric descriptors.
    if np.max(np.sum(cm, axis=1)) >= threshold:

        class_id = np.argmax(cm[:, feature])

        # Masking classes based on the criterias set above. Mask will present
        # True for wood points and False for leaf points.
        mask = classes == class_id

    else:
        mask = []

    return mask
