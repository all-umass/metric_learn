"""
Covariance metric (baseline method)

This method does not "learn" anything, rather it calculates
the covariance matrix of the input data.

This is a simple baseline method first introduced in
On the Generalized Distance in Statistics, P.C.Mahalanobis, 1936
"""

from __future__ import absolute_import
import numpy as np
from sklearn.utils.validation import check_array
from sklearn.base import TransformerMixin

from .base_metric import MahalanobisMixin


class Covariance(MahalanobisMixin, TransformerMixin):
  """Covariance metric (baseline method)

  Attributes
  ----------
  transformer_ : `numpy.ndarray`, shape=(num_dims, n_features)
      The linear transformation ``L`` deduced from the learned Mahalanobis
      metric (See :meth:`transformer_from_metric`.)
  """

  def __init__(self):
    pass

  def fit(self, X, y=None):
    """
    X : data matrix, (n x d)
    y : unused
    """
    self.X_ = check_array(X, ensure_min_samples=2)
    self.M_ = np.cov(self.X_, rowvar = False)
    if self.M_.ndim == 0:
      self.M_ = 1./self.M_
    else:
      self.M_ = np.linalg.inv(self.M_)

    self.transformer_ = self.transformer_from_metric(check_array(self.M_))
    return self
