"""
Large Margin Nearest Neighbor Metric learning (LMNN)
"""
import numpy as np
from collections import Counter
from sklearn.metrics import euclidean_distances
from sklearn.base import TransformerMixin
import warnings

from ._util import _initialize_components, _check_n_components
from .base_metric import MahalanobisMixin


class LMNN(MahalanobisMixin, TransformerMixin):
  """Large Margin Nearest Neighbor (LMNN)

  LMNN learns a Mahalanobis distance metric in the kNN classification
  setting. The learned metric attempts to keep close k-nearest neighbors
  from the same class, while keeping examples from different classes
  separated by a large margin. This algorithm makes no assumptions about
  the distribution of the data.

  Read more in the :ref:`User Guide <lmnn>`.

  Parameters
  ----------
  init : string or numpy array, optional (default='auto')
    Initialization of the linear transformation. Possible options are
    'auto', 'pca', 'identity', 'random', and a numpy array of shape
    (n_features_a, n_features_b).

    'auto'
      Depending on ``n_components``, the most reasonable initialization
      will be chosen. If ``n_components <= n_classes`` we use 'lda', as
      it uses labels information. If not, but
      ``n_components < min(n_features, n_samples)``, we use 'pca', as
      it projects data in meaningful directions (those of higher
      variance). Otherwise, we just use 'identity'.

    'pca'
      ``n_components`` principal components of the inputs passed
      to :meth:`fit` will be used to initialize the transformation.
      (See `sklearn.decomposition.PCA`)

    'lda'
      ``min(n_components, n_classes)`` most discriminative
      components of the inputs passed to :meth:`fit` will be used to
      initialize the transformation. (If ``n_components > n_classes``,
      the rest of the components will be zero.) (See
      `sklearn.discriminant_analysis.LinearDiscriminantAnalysis`)

    'identity'
      If ``n_components`` is strictly smaller than the
      dimensionality of the inputs passed to :meth:`fit`, the identity
      matrix will be truncated to the first ``n_components`` rows.

    'random'
      The initial transformation will be a random array of shape
      `(n_components, n_features)`. Each value is sampled from the
      standard normal distribution.

    numpy array
      n_features_b must match the dimensionality of the inputs passed to
      :meth:`fit` and n_features_a must be less than or equal to that.
      If ``n_components`` is not None, n_features_a must match it.

  n_neighbors : int, optional (default=3)
    Number of neighbors to consider, not including self-edges.

  min_iter : int, optional (default=50)
    Minimum number of iterations of the optimization procedure.

  max_iter : int, optional (default=1000)
    Maximum number of iterations of the optimization procedure.

  learn_rate : float, optional (default=1e-7)
    Learning rate of the optimization procedure

  tol : float, optional (default=0.001)
    Tolerance of the optimization procedure. If the objective value varies
    less than `tol`, we consider the algorithm has converged and stop it.

  verbose : bool, optional (default=False)
    Whether to print the progress of the optimization procedure.

  regularization: float, optional (default=0.5)
    Relative weight between pull and push terms, with 0.5 meaning equal
    weight.

  preprocessor : array-like, shape=(n_samples, n_features) or callable
    The preprocessor to call to get tuples from indices. If array-like,
    tuples will be formed like this: X[indices].

  n_components : int or None, optional (default=None)
    Dimensionality of reduced space (if None, defaults to dimension of X).

  random_state : int or numpy.RandomState or None, optional (default=None)
    A pseudo random number generator object or a seed for it if int. If
    ``init='random'``, ``random_state`` is used to initialize the random
    transformation. If ``init='pca'``, ``random_state`` is passed as an
    argument to PCA when initializing the transformation.

  k : Renamed to n_neighbors. Will be deprecated in 0.7.0

  Attributes
  ----------
  n_iter_ : `int`
    The number of iterations the solver has run.

  components_ : `numpy.ndarray`, shape=(n_components, n_features)
    The learned linear transformation ``L``.

  Examples
  --------

  >>> import numpy as np
  >>> from metric_learn import LMNN
  >>> from sklearn.datasets import load_iris
  >>> iris_data = load_iris()
  >>> X = iris_data['data']
  >>> Y = iris_data['target']
  >>> lmnn = LMNN(n_neighbors=5, learn_rate=1e-6)
  >>> lmnn.fit(X, Y, verbose=False)

  References
  ----------
  .. [1] K. Q. Weinberger, J. Blitzer, L. K. Saul. `Distance Metric
         Learning for Large Margin Nearest Neighbor Classification
         <http://papers.nips.cc/paper/2795-distance-metric\
         -learning-for-large-margin-nearest-neighbor-classification>`_. NIPS
         2005.
  """

  def __init__(self, init='auto', n_neighbors=3, min_iter=50, max_iter=1000,
               learn_rate=1e-7, regularization=0.5, convergence_tol=0.001,
               verbose=False, preprocessor=None,
               n_components=None, random_state=None, k='deprecated'):
    self.init = init
    if k != 'deprecated':
      warnings.warn('"num_chunks" parameter has been renamed to'
                    ' "n_chunks". It has been deprecated in'
                    ' version 0.6.3 and will be removed in 0.7.0'
                    '', FutureWarning)
      n_neighbors = k
    self.k = 'deprecated'  # To avoid no_attribute error
    self.n_neighbors = n_neighbors
    self.min_iter = min_iter
    self.max_iter = max_iter
    self.learn_rate = learn_rate
    self.regularization = regularization
    self.convergence_tol = convergence_tol
    self.verbose = verbose
    self.n_components = n_components
    self.random_state = random_state
    super(LMNN, self).__init__(preprocessor)

  def fit(self, X, y):
    k = self.n_neighbors
    reg = self.regularization
    learn_rate = self.learn_rate

    X, y = self._prepare_inputs(X, y, dtype=float,
                                ensure_min_samples=2)
    num_pts, d = X.shape
    output_dim = _check_n_components(d, self.n_components)
    unique_labels, label_inds = np.unique(y, return_inverse=True)
    if len(label_inds) != num_pts:
      raise ValueError('Must have one label per point.')
    self.labels_ = np.arange(len(unique_labels))

    self.components_ = _initialize_components(output_dim, X, y, self.init,
                                              self.verbose,
                                              random_state=self.random_state)
    required_k = np.bincount(label_inds).min()
    if self.n_neighbors >= required_k:
      raise ValueError('not enough class labels for specified k'
                       ' (smallest class has %d)' % required_k)

    target_neighbors = self._select_targets(X, label_inds)

    # sum outer products
    dfG = _sum_outer_products(X, target_neighbors.flatten(),
                              np.repeat(np.arange(X.shape[0]), k))

    # initialize L
    L = self.components_

    # first iteration: we compute variables (including objective and gradient)
    #  at initialization point
    G, objective, total_active = self._loss_grad(X, L, dfG, k,
                                                 reg, target_neighbors,
                                                 label_inds)

    it = 1  # we already made one iteration

    if self.verbose:
      print("iter | objective | objective difference | active constraints",
            "| learning rate")

    # main loop
    for it in range(2, self.max_iter):
      # then at each iteration, we try to find a value of L that has better
      # objective than the previous L, following the gradient:
      while True:
        # the next point next_L to try out is found by a gradient step
        L_next = L - learn_rate * G
        # we compute the objective at next point
        # we copy variables that can be modified by _loss_grad, because if we
        # retry we don t want to modify them several times
        (G_next, objective_next, total_active_next) = (
            self._loss_grad(X, L_next, dfG, k, reg, target_neighbors,
                            label_inds))
        assert not np.isnan(objective)
        delta_obj = objective_next - objective
        if delta_obj > 0:
          # if we did not find a better objective, we retry with an L closer to
          # the starting point, by decreasing the learning rate (making the
          # gradient step smaller)
          learn_rate /= 2
        else:
          # otherwise, if we indeed found a better obj, we get out of the loop
          break
      # when the better L is found (and the related variables), we set the
      # old variables to these new ones before next iteration and we
      # slightly increase the learning rate
      L = L_next
      G, objective, total_active = G_next, objective_next, total_active_next
      learn_rate *= 1.01

      if self.verbose:
        print(it, objective, delta_obj, total_active, learn_rate)

      # check for convergence
      if it > self.min_iter and abs(delta_obj) < self.convergence_tol:
        if self.verbose:
          print("LMNN converged with objective", objective)
        break
    else:
      if self.verbose:
        print("LMNN didn't converge in %d steps." % self.max_iter)

    # store the last L
    self.components_ = L
    self.n_iter_ = it
    return self

  def _loss_grad(self, X, L, dfG, k, reg, target_neighbors, label_inds):
    # Compute pairwise distances under current metric
    Lx = L.dot(X.T).T

    # we need to find the furthest neighbor:
    Ni = 1 + _inplace_paired_L2(Lx[target_neighbors], Lx[:, None, :])
    furthest_neighbors = np.take_along_axis(target_neighbors,
                                            Ni.argmax(axis=1)[:, None], 1)
    impostors = self._find_impostors(furthest_neighbors.ravel(), X,
                                     label_inds, L)

    g0 = _inplace_paired_L2(*Lx[impostors])

    # we reorder the target neighbors
    g1, g2 = Ni[impostors]
    # compute the gradient
    total_active = 0
    df = np.zeros((X.shape[1], X.shape[1]))
    for nn_idx in reversed(range(k)):  # note: reverse not useful here
      act1 = g0 < g1[:, nn_idx]
      act2 = g0 < g2[:, nn_idx]
      total_active += act1.sum() + act2.sum()

      targets = target_neighbors[:, nn_idx]
      PLUS, pweight = _count_edges(act1, act2, impostors, targets)
      df += _sum_outer_products(X, PLUS[:, 0], PLUS[:, 1], pweight)

      in_imp, out_imp = impostors
      df -= _sum_outer_products(X, in_imp[act1], out_imp[act1])
      df -= _sum_outer_products(X, in_imp[act2], out_imp[act2])

    # do the gradient update
    assert not np.isnan(df).any()
    G = dfG * reg + df * (1 - reg)
    G = L.dot(G)
    # compute the objective function
    objective = total_active * (1 - reg)
    objective += G.flatten().dot(L.flatten())
    return 2 * G, objective, total_active

  def _select_targets(self, X, label_inds):
    target_neighbors = np.empty((X.shape[0], self.n_neighbors), dtype=int)
    for label in self.labels_:
      inds, = np.nonzero(label_inds == label)
      dd = euclidean_distances(X[inds], squared=True)
      np.fill_diagonal(dd, np.inf)
      nn = np.argsort(dd)[..., :self.n_neighbors]
      target_neighbors[inds] = inds[nn]
    return target_neighbors

  def _find_impostors(self, furthest_neighbors, X, label_inds, L):
    Lx = X.dot(L.T)
    margin_radii = 1 + _inplace_paired_L2(Lx[furthest_neighbors], Lx)
    impostors = []
    for label in self.labels_[:-1]:
      in_inds, = np.nonzero(label_inds == label)
      out_inds, = np.nonzero(label_inds > label)
      dist = euclidean_distances(Lx[out_inds], Lx[in_inds], squared=True)
      i1, j1 = np.nonzero(dist < margin_radii[out_inds][:, None])
      i2, j2 = np.nonzero(dist < margin_radii[in_inds])
      i = np.hstack((i1, i2))
      j = np.hstack((j1, j2))
      if i.size > 0:
        # get unique (i,j) pairs using index trickery
        shape = (i.max() + 1, j.max() + 1)
        tmp = np.ravel_multi_index((i, j), shape)
        i, j = np.unravel_index(np.unique(tmp), shape)
      impostors.append(np.vstack((in_inds[j], out_inds[i])))
    if len(impostors) == 0:
        # No impostors detected
        return impostors
    return np.hstack(impostors)


def _inplace_paired_L2(A, B):
  '''Equivalent to ((A-B)**2).sum(axis=-1), but modifies A in place.'''
  A -= B
  return np.einsum('...ij,...ij->...i', A, A)


def _count_edges(act1, act2, impostors, targets):
  imp = impostors[0, act1]
  c = Counter(zip(imp, targets[imp]))
  imp = impostors[1, act2]
  c.update(zip(imp, targets[imp]))
  if c:
    active_pairs = np.array(list(c.keys()))
  else:
    active_pairs = np.empty((0, 2), dtype=int)
  return active_pairs, np.array(list(c.values()))


def _sum_outer_products(data, a_inds, b_inds, weights=None):
  Xab = data[a_inds] - data[b_inds]
  if weights is not None:
    return np.dot(Xab.T, Xab * weights[:, None])
  return np.dot(Xab.T, Xab)
