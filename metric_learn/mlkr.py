"""
Metric Learning for Kernel Regression (MLKR), Weinberger et al.,

MLKR is an algorithm for supervised metric learning, which learns a distance
function by directly minimising the leave-one-out regression error. This
algorithm can also be viewed as a supervised variation of PCA and can be used
for dimensionality reduction and high dimensional data visualization.
"""
from __future__ import division
import numpy as np
from six.moves import xrange
from scipy.spatial.distance import pdist, squareform

from .base_metric import BaseMetricLearner

class MLKR(BaseMetricLearner):
    """Metric Learning for Kernel Regression (MLKR)"""
    def __init__(self, A0=None, epsilon=0.01, alpha=0.0001):
        """
        MLKR initialization

        Parameters
        ----------
        A0: Initialization of matrix A. Defaults to the identity matrix.
        epsilon: Step size for gradient descent.
        alpha: Stopping criterion for loss function in gradient descent.
        """
        self.params = {
            "A0": A0,
            "epsilon": epsilon,
            "alpha": alpha
        }

    def _process_inputs(self, X, y):
        self.X = np.array(X, copy=False)
        y = np.array(y, copy=False)
        if X.ndim == 1:
            X = X[:, np.newaxis]
        if y.ndim == 1:
            y = y[:, np.newaxis]
        n, d = X.shape
        if y.shape[0] != n:
            raise ValueError('Data and label lengths mismatch: %d != %d'
                             % (n, y.shape[0]))
        return y, n, d

    def fit(self, X, y):
        """
        Fit MLKR model

        Parameters:
        ----------
        X : (n x d) array of samples
        y : (n) data labels

        Returns:
        -------
        self: Instance of self
        """
        y, n, d = self._process_inputs(X, y)
        if self.params['A0'] is None:
            A = np.identity(d)  # Initialize A as eye matrix
        else:
            A = self.params['A0']
            if A.shape != (d, d):
                raise ValueError('A0 should be a square matrix of dimension'
                                 ' %d. %s shape was provided' % (d, A.shape))
        cost = np.Inf
        # Gradient descent procedure
        alpha = self.params['alpha']
        epsilon = self.params['epsilon']
        while cost > alpha:
            K = self._computeK(X, A)
            yhat = self._computeyhat(y, K)
            cost = np.sum(np.square(yhat - y))
            # Compute gradient
            sum_i = 0
            for i in xrange(n):
                sum_j = 0
                for j in xrange(n):
                    diffK = (yhat[j] - y[j]) * K[i, j]
                    x_ij = (X[i, :] - X[j, :])[:, np.newaxis]
                    x_ijT = x_ij.T
                    sum_j += diffK * x_ij.dot(x_ijT)
                sum_i += (yhat[i] - y[i]) * sum_j
            gradient = 4 * A.dot(sum_i)
            A -= epsilon * gradient
        self._transformer = A
        return self

    @staticmethod
    def _computeK(X, A):
        """
        Internal helper function to compute K matrix.

        Parameters:
        ----------
        X: (n x d) array of samples
        A: (d x d) 'A' matrix

        Returns:
        -------
        K: (n x n) K matrix where Kij = exp(-distance(x_i, x_j)) where
           distance is defined as squared L2 norm of (x_i - x_j)
        """
        dist_mat = pdist(X, metric='mahalanobis', VI=A.T.dot(A))
        return np.exp(squareform(-(dist_mat ** 2)))

    @staticmethod
    def _computeyhat(y, K):
        """
        Internal helper function to compute yhat matrix.

        Parameters:
        ----------
        y: (n) data labels
        K: (n x n) K matrix

        Returns:
        -------
        yhat: (n x 1) yhat matrix
        """
        K_mod = np.copy(K)
        np.fill_diagonal(K_mod, 0)
        numerator = K_mod.dot(y)
        denominator = np.sum(K_mod, 1)[:, np.newaxis]
        denominator[denominator == 0] = 2.2204e-16  # eps val in octave
        yhat = numerator / denominator
        return yhat

    def transformer(self):
        return self._transformer
