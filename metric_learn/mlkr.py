"""
Metric Learning for Kernel Regression (MLKR), Weinberger et al.,

MLKR is an algorithm for supervised metric learning, which learns a distance
function by directly minimising the leave-one-out regression error. This
algorithm can also be viewed as a supervised variation of PCA and can be used
for dimensionality reduction and high dimensional data visualization.
"""
from __future__ import division, print_function
import time
import sys
import warnings
import numpy as np
from sklearn.exceptions import ConvergenceWarning
from sklearn.utils.fixes import logsumexp
from scipy.optimize import minimize
from scipy.spatial.distance import pdist, squareform
from sklearn.base import TransformerMixin
from sklearn.decomposition import PCA


from sklearn.metrics import pairwise_distances
from .base_metric import MahalanobisMixin
from metric_learn._util import _initialize_transformer

EPS = np.finfo(float).eps


class MLKR(MahalanobisMixin, TransformerMixin):
  """Metric Learning for Kernel Regression (MLKR)

  Attributes
  ----------
  n_iter_ : `int`
      The number of iterations the solver has run.

  transformer_ : `numpy.ndarray`, shape=(num_dims, n_features)
      The learned linear transformation ``L``.
  """

  def __init__(self, num_dims=None, init='auto', A0=None,
               tol=None, max_iter=1000, verbose=False, preprocessor=None,
               random_state=None):
    """
    Initialize MLKR.

    Parameters
    ----------
    num_dims : int, optional
        Dimensionality of reduced space (defaults to dimension of X)

    init : string or numpy array, optional (default='auto')
        Initialization of the linear transformation. Possible options are
        'auto', 'pca', 'lda', 'identity', 'random', and a numpy array of shape
        (n_features_a, n_features_b).

        'auto'
            Depending on ``num_dims``, the most reasonable initialization
            will be chosen. If ``num_dims < min(n_features, n_samples)``, we
            use 'pca', as it projects data in meaningful directions (those
            of higher variance). Otherwise, we just use 'identity'.

        'pca'
            ``num_dims`` principal components of the inputs passed
            to :meth:`fit` will be used to initialize the transformation.
            (See `sklearn.decomposition.PCA`)

        'identity'
            If ``num_dims`` is strictly smaller than the
            dimensionality of the inputs passed to :meth:`fit`, the identity
            matrix will be truncated to the first ``num_dims`` rows.

        'random'
            The initial transformation will be a random array of shape
            `(num_dims, n_features)`. Each value is sampled from the
            standard normal distribution.

        numpy array
            n_features_b must match the dimensionality of the inputs passed to
            :meth:`fit` and n_features_a must be less than or equal to that.
            If ``num_dims`` is not None, n_features_a must match it.

    A0: array-like, optional # TODO: deprecate
        Initialization of transformation matrix. Defaults to PCA loadings.

    tol: float, optional (default=None)
        Convergence tolerance for the optimization.

    max_iter: int, optional
        Cap on number of congugate gradient iterations.

    verbose : bool, optional (default=False)
        Whether to print progress messages or not.

    preprocessor : array-like, shape=(n_samples, n_features) or callable
        The preprocessor to call to get tuples from indices. If array-like,
        tuples will be formed like this: X[indices].

    random_state : int or numpy.RandomState or None, optional (default=None)
        A pseudo random number generator object or a seed for it if int. If
        ``init='random'``, ``random_state`` is used to initialize the random
        transformation. If ``init='pca'``, ``random_state`` is passed as an
        argument to PCA when initializing the transformation.
    """
    self.num_dims = num_dims
    self.init = init
    self.A0 = A0  # TODO: deprecate
    self.tol = tol
    self.max_iter = max_iter
    self.verbose = verbose
    self.random_state = random_state
    super(MLKR, self).__init__(preprocessor)

  def fit(self, X, y):
      """
      Fit MLKR model

      Parameters
      ----------
      X : (n x d) array of samples
      y : (n) data labels
      """
      X, y = self._prepare_inputs(X, y, y_numeric=True,
                                  ensure_min_samples=2)
      n, d = X.shape
      if y.shape[0] != n:
          raise ValueError('Data and label lengths mismatch: %d != %d'
                           % (n, y.shape[0]))

      m = self.num_dims
      if m is None:
          m = d
      A = _initialize_transformer(m, X, y, init=self.init,
                                  random_state=self.random_state)

      # Measure the total training time
      train_time = time.time()

      self.n_iter_ = 0
      res = minimize(self._loss, A.ravel(), (X, y), method='L-BFGS-B',
                     jac=True, tol=self.tol,
                     options=dict(maxiter=self.max_iter))
      self.transformer_ = res.x.reshape(A.shape)

      # Stop timer
      train_time = time.time() - train_time
      if self.verbose:
          cls_name = self.__class__.__name__
          # Warn the user if the algorithm did not converge
          if not res.success:
              warnings.warn('[{}] MLKR did not converge: {}'
                            .format(cls_name, res.message), ConvergenceWarning)
          print('[{}] Training took {:8.2f}s.'.format(cls_name, train_time))

      return self

  def _loss(self, flatA, X, y):

    if self.n_iter_ == 0 and self.verbose:
      header_fields = ['Iteration', 'Objective Value', 'Time(s)']
      header_fmt = '{:>10} {:>20} {:>10}'
      header = header_fmt.format(*header_fields)
      cls_name = self.__class__.__name__
      print('[{cls}]'.format(cls=cls_name))
      print('[{cls}] {header}\n[{cls}] {sep}'.format(cls=cls_name,
                                                     header=header,
                                                     sep='-' * len(header)))

    start_time = time.time()

    A = flatA.reshape((-1, X.shape[1]))
    X_embedded = np.dot(X, A.T)
    dist = pairwise_distances(X_embedded, squared=True)
    np.fill_diagonal(dist, np.inf)
    softmax = np.exp(- dist - logsumexp(- dist, axis=1)[:, np.newaxis])
    yhat = softmax.dot(y)
    ydiff = yhat - y
    cost = (ydiff ** 2).sum()

    # also compute the gradient
    W = softmax * ydiff[:, np.newaxis] * (y - yhat[:, np.newaxis])
    W_sym = W + W.T
    np.fill_diagonal(W_sym, - W.sum(axis=0))
    grad = 4 * (X_embedded.T.dot(W_sym)).dot(X)

    if self.verbose:
      start_time = time.time() - start_time
      values_fmt = '[{cls}] {n_iter:>10} {loss:>20.6e} {start_time:>10.2f}'
      print(values_fmt.format(cls=self.__class__.__name__,
                              n_iter=self.n_iter_, loss=cost,
                              start_time=start_time))
      sys.stdout.flush()

    self.n_iter_ += 1

    return cost, grad.ravel()
