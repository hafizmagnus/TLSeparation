# -*- coding: utf-8 -*-
"""
Module to manage the Gaussian Mixture Model classification of a dataset.

@author: Matheus Boni Vicari (matheus.boni.vicari@gmail.com)
"""

from sklearn.mixture import GaussianMixture as GMM


def classify(variables, n_classes):

    """
    Function to perform the classification of a dataset using sklearn's
    Gaussian Mixture Models with Expectation Maximization.

    Parameters
    ----------
    variables: array_like
        N-dimensional array (m x n) containing a set of parameters (n) over
        a set of observations (m).
    n_classes: int
        Number of classes to assign the input variables.

    Returns
    -------
    classes: list
        List of classes labels for each observation from the input variables.
    means: array
        N-dimensional array (c x n) of each class (c) parameter space means
        (n).
    probability: array
        Probability of samples belonging to every class in the classification.
        Sum of sample-wise probability should be 1.

    """

    gmm = GMM(n_components=n_classes)
    gmm.fit(variables)
    return gmm.predict(variables), gmm.means_, gmm.predict_proba(variables)
