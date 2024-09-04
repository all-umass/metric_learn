"""
Sparse Compositional Metric Learning (SCML)
"""

from __future__ import print_function, absolute_import, division
import numpy as np
from .base_metric import _TripletsClassifierMixin, MahalanobisMixin
from ._util import components_from_metric
from sklearn.base import TransformerMixin
from .constraints import Constraints
from sklearn.preprocessing import normalize
from sklearn.neighbors import NearestNeighbors
from sklearn.cluster import KMeans
from sklearn.discriminant_analysis import LinearDiscriminantAnalysis
from sklearn.utils import check_array, check_random_state
import warnings


class _BaseSCML(MahalanobisMixin):

  _tuple_size = 3   # constraints are triplets
  _authorized_basis = ['triplet_diffs']

  def __init__(self, beta=1e-5, basis='triplet_diffs', n_basis=None,
               gamma=5e-3, max_iter=10000, output_iter=500, batch_size=10,
               verbose=False, preprocessor=None, random_state=None,
               warm_start=False):
    self.beta = beta
    self.basis = basis
    self.n_basis = n_basis
    self.gamma = gamma
    self.max_iter = max_iter
    self.output_iter = output_iter
    self.batch_size = batch_size
    self.verbose = verbose
    self.preprocessor = preprocessor
    self.random_state = random_state
    self.warm_start = warm_start
    super(_BaseSCML, self).__init__(preprocessor)

  def _fit(self, triplets, basis=None, n_basis=None):
    """
    Optimization procedure to find a sparse vector of weights to
    construct the metric from the basis set. This is based on the
    dual averaging method.
    """

    if not isinstance(self.max_iter, int):
      raise ValueError("max_iter should be an integer, instead it is of type"
                       " %s" % type(self.max_iter))
    if not isinstance(self.output_iter, int):
      raise ValueError("output_iter should be an integer, instead it is of "
                       "type %s" % type(self.output_iter))
    if not isinstance(self.batch_size, int):
      raise ValueError("batch_size should be an integer, instead it is of type"
                       " %s" % type(self.batch_size))

    if self.output_iter > self.max_iter:
      raise ValueError("The value of output_iter must be equal or smaller than"
                       " max_iter.")

    # Currently prepare_inputs makes triplets contain points and not indices
    triplets = self._prepare_inputs(triplets, type_of_inputs='tuples')

    # TODO:
    # This algorithm is built to work with indices, but in order to be
    # compliant with the current handling of inputs it is converted
    # back to indices by the following function. This should be improved
    # in the future.
    triplets, X = self._to_index_points(triplets)

    if basis is None:
      basis, n_basis = self._initialize_basis(triplets, X)

    dist_diff = self._compute_dist_diff(triplets, X, basis)

    n_triplets = triplets.shape[0]

    if not self.warm_start or not hasattr(self, "w_"):
        # weight vector
        self.w_ = np.zeros((1, n_basis))
        # average obj gradient wrt weights
        self.avg_grad_w_ = np.zeros((1, n_basis))
        # l2 norm in time of all obj gradients wrt weights
        self.ada_grad_w_ = np.zeros((1, n_basis))

    # slack for not dividing by zero
    delta = 0.001

    best_obj = np.inf

    rng = check_random_state(self.random_state)
    rand_int = rng.randint(low=0, high=n_triplets,
                           size=(self.max_iter, self.batch_size))
    for iter in range(self.max_iter):

      idx = rand_int[iter]

      slack_val = 1 + np.matmul(dist_diff[idx, :], self.w_.T)
      slack_mask = np.squeeze(slack_val > 0, axis=1)

      grad_w = np.sum(dist_diff[idx[slack_mask], :],
                      axis=0, keepdims=True)/self.batch_size

      self.avg_grad_w_ = (iter * self.avg_grad_w_ + grad_w) / (iter+1)

      self.ada_grad_w_ = np.sqrt(np.square(self.ada_grad_w_) +
                                 np.square(grad_w))

      scale_f = -(iter+1) / (self.gamma * (delta + self.ada_grad_w_))

      # proximal operator with negative trimming equivalent
      self.w_ = scale_f * np.minimum(self.avg_grad_w_ + self.beta, 0)

      if (iter + 1) % self.output_iter == 0:
        # regularization part of obj function
        obj1 = np.sum(self.w_)*self.beta

        # Every triplet distance difference in the space given by L
        # plus a slack of one
        slack_val = 1 + np.matmul(dist_diff, self.w_.T)
        # Mask of places with positive slack
        slack_mask = slack_val > 0

        # loss function of learning task part of obj function
        obj2 = np.sum(slack_val[slack_mask])/n_triplets

        obj = obj1 + obj2
        if self.verbose:
          count = np.sum(slack_mask)
          print("[%s] iter %d\t obj %.6f\t num_imp %d" %
                (self.__class__.__name__, (iter+1), obj, count))

        # update the best
        if obj < best_obj:
          best_obj = obj
          best_w = self.w_

    if self.verbose:
      print("max iteration reached.")

    # return L matrix yielded from best weights
    self.n_iter_ = iter
    self.components_ = self._components_from_basis_weights(basis, best_w)

    return self

  def _compute_dist_diff(self, triplets, X, basis):
    """
    Helper function to compute the distance difference of every triplet in the
    space yielded by the basis set.
    """
    # Transformation of data by the basis set
    XB = np.matmul(X, basis.T)

    n_triplets = triplets.shape[0]
    # get all positive and negative pairs with lowest index first
    # np.array (2*n_triplets,2)
    triplets_pairs_sorted = np.sort(np.vstack((triplets[:, [0, 1]],
                                               triplets[:, [0, 2]])),
                                    kind='stable')
    # calculate all unique pairs and their indices
    uniqPairs, indices = np.unique(triplets_pairs_sorted, return_inverse=True,
                                   axis=0)
    # calculate L2 distance acording to bases only for unique pairs
    dist = np.square(XB[uniqPairs[:, 0], :] - XB[uniqPairs[:, 1], :])

    # return the diference of distances between all positive and negative
    # pairs
    return dist[indices[:n_triplets]] - dist[indices[n_triplets:]]

  def _components_from_basis_weights(self, basis, w):
    """
    Get components matrix (L) from computed mahalanobis matrix.
    """

    # get rid of inactive bases
    # TODO: Maybe have a tolerance over zero?
    active_idx, = w > 0
    w = w[..., active_idx]
    basis = basis[active_idx, :]

    n_basis, n_features = basis.shape

    if n_basis < n_features:  # if metric is low-rank
      warnings.warn("The number of bases with nonzero weight is less than the "
                    "number of features of the input, in consequence the "
                    "learned transformation reduces the dimension to %d."
                    % n_basis)
      return np.sqrt(w.T)*basis  # equivalent to np.diag(np.sqrt(w)).dot(basis)

    else:   # if metric is full rank
      return components_from_metric(np.matmul(basis.T, w.T*basis))

  def _to_index_points(self, triplets):
    shape = triplets.shape
    X, triplets = np.unique(np.vstack(triplets), return_inverse=True, axis=0)
    triplets = triplets.reshape(shape[:2])
    return triplets, X

  def _initialize_basis(self, triplets, X):
    """ Checks if the basis array is well constructed or constructs it based
    on one of the available options.
    """
    n_features = X.shape[1]

    if isinstance(self.basis, np.ndarray):
      # TODO: should copy?
      basis = check_array(self.basis, copy=True)
      if basis.shape[1] != n_features:
        raise ValueError('The dimensionality ({}) of the provided bases must'
                         ' match the dimensionality of the data '
                         '({}).'.format(basis.shape[1], n_features))
    elif self.basis not in self._authorized_basis:
      raise ValueError(
          "`basis` must be one of the options '{}' "
          "or an array of shape (n_basis, n_features)."
          .format("', '".join(self._authorized_basis)))
    if self.basis == 'triplet_diffs':
      basis, n_basis = self._generate_bases_dist_diff(triplets, X)

    return basis, n_basis

  def _generate_bases_dist_diff(self, triplets, X):
    """ Constructs the basis set from the differences of positive and negative
    pairs from the triplets constraints.

    The basis set is constructed iteratively by taking n_features triplets,
    then adding and substracting respectively all the outerproducts of the
    positive and negative pairs, and finally selecting the eigenvectors
    of this matrix with positive eigenvalue. This is done until n_basis are
    selected.
    """
    n_features = X.shape[1]
    n_triplets = triplets.shape[0]

    if self.n_basis is None:
      # TODO: Get a good default n_basis directive
      n_basis = n_features*80
      warnings.warn('As no value for `n_basis` was selected, the number of '
                    'basis will be set to n_basis= %d' % n_basis)
    elif isinstance(self.n_basis, int):
      n_basis = self.n_basis
    else:
      raise ValueError("n_basis should be an integer, instead it is of type %s"
                       % type(self.n_basis))

    if n_features > n_triplets:
      raise ValueError(
        "Number of features (%s) is greater than the number of triplets(%s).\n"
        "Consider using dimensionality reduction or using another basis "
        "generation scheme." % (n_features, n_triplets))

    basis = np.zeros((n_basis, n_features))

    # get all positive and negative pairs with lowest index first
    # np.array (2*n_triplets,2)
    triplets_pairs_sorted = np.sort(np.vstack((triplets[:, [0, 1]],
                                               triplets[:, [0, 2]])),
                                    kind='stable')
    # calculate all unique pairs and their indices
    uniqPairs, indices = np.unique(triplets_pairs_sorted, return_inverse=True,
                                   axis=0)
    # calculate differences only for unique pairs
    diff = X[uniqPairs[:, 0], :] - X[uniqPairs[:, 1], :]

    diff_pos = diff[indices[:n_triplets], :]
    diff_neg = diff[indices[n_triplets:], :]

    rng = check_random_state(self.random_state)

    start = 0
    finish = 0
    while finish != n_basis:
      # Select triplets to yield diff
      select_triplet = rng.choice(n_triplets, size=n_features, replace=False)

      # select n_features positive differences
      d_pos = diff_pos[select_triplet, :]

      # select n_features negative differences
      d_neg = diff_neg[select_triplet, :]

      # Yield matrix
      diff_sum = d_pos.T.dot(d_pos) - d_neg.T.dot(d_neg)

      # Calculate eigenvalue and eigenvectors
      w, v = np.linalg.eigh(diff_sum.T.dot(diff_sum))

      # Add eigenvectors with positive eigenvalue to basis set
      pos_eig_mask = w > 0
      start = finish
      finish += pos_eig_mask.sum()

      try:
        basis[start:finish, :] = v[pos_eig_mask]
      except ValueError:
        # if finish is greater than n_basis
        basis[start:, :] = v[pos_eig_mask][:n_basis-start]
        break

      # TODO: maybe add a warning in case there are no added bases, this could
      # be caused by a bad triplet set. This would cause an infinite loop

    return basis, n_basis


class SCML(_BaseSCML, _TripletsClassifierMixin):
  """Sparse Compositional Metric Learning (SCML)

  `SCML` learns an squared Mahalanobis distance from triplet constraints by
  optimizing sparse positive weights assigned to a set of :math:`K` rank-one
  PSD bases. This can be formulated as an optimization problem with only
  :math:`K` parameters, that can be solved with an efficient stochastic
  composite scheme.

  Read more in the :ref:`User Guide <scml>`.

  .. warning::
    SCML is still a bit experimental, don't hesitate to report if
    something fails/doesn't work as expected.

  Parameters
  ----------
  beta: float (default=1e-5)
    L1 regularization parameter.

  basis : string or array-like, optional (default='triplet_diffs')
    Set of bases to construct the metric. Possible options are
    'triplet_diffs', and an array-like of shape (n_basis, n_features).

    'triplet_diffs'
      The basis set is constructed iteratively from differences between points
      of `n_features` positive or negative pairs randomly sampled from the
      triplets constraints. Requires the number of training triplets to be
      great or equal to `n_features`.

    array-like
        A matrix of shape (n_basis, n_features), that will be used as
        the basis set for the metric construction.

  n_basis : int, optional
    Number of basis to be yielded. In case it is not set it will be set based
    on `basis`. If no value is selected a default will be computed based on
    the input.

  gamma: float (default = 5e-3)
    Learning rate for the optimization algorithm.

  max_iter : int (default = 10000)
    Number of iterations for the algorithm.

  output_iter : int (default = 5000)
    Number of iterations to check current weights performance and output this
    information in case verbose is True.

  verbose : bool, optional
    If True, prints information while learning.

  preprocessor : array-like, shape=(n_samples, n_features) or callable
    The preprocessor to call to get triplets from indices. If array-like,
    triplets will be formed like this: X[indices].

  random_state : int or numpy.RandomState or None, optional (default=None)
    A pseudo random number generator object or a seed for it if int.

  warm_start : bool, default=False
    When set to True, reuse the solution of the previous call to fit as
    initialization, otherwise, just erase the previous solution.
    Repeatedly calling fit when warm_start is True can result in a different
    solution than when calling fit a single time because of the way the data
    is shuffled.

  Attributes
  ----------
  components_ : `numpy.ndarray`, shape=(n_features, n_features)
    The linear transformation ``L`` deduced from the learned Mahalanobis
    metric (See function `_components_from_basis_weights`.)

  Examples
  --------
  >>> from metric_learn import SCML
  >>> triplets = [[[1.2, 7.5], [1.3, 1.5], [6.2, 9.7]],
  >>>             [[1.3, 4.5], [3.2, 4.6], [5.4, 5.4]],
  >>>             [[3.2, 7.5], [3.3, 1.5], [8.2, 9.7]],
  >>>             [[3.3, 4.5], [5.2, 4.6], [7.4, 5.4]]]
  >>> scml = SCML()
  >>> scml.fit(triplets)

  References
  ----------
  .. [1] Y. Shi, A. Bellet and F. Sha. `Sparse Compositional Metric Learning.
         <http://researchers.lille.inria.fr/abellet/papers/aaai14.pdf>`_. \
         (AAAI), 2014.

  .. [2] Adapted from original `Matlab implementation. \
         <https://github.com/bellet/SCML>`_.

  See Also
  --------
  metric_learn.SCML_Supervised : The supervised version of the algorithm.

  :ref:`supervised_version` : The section of the project documentation
    that describes the supervised version of weakly supervised estimators.
  """

  def fit(self, triplets):
    """Learn the SCML model.

    Parameters
    ----------
    triplets : array-like, shape=(n_constraints, 3, n_features) or \
      (n_constraints, 3)
      3D array-like of triplets of points or 2D array of triplets of
      indicators. Triplets are assumed to be ordered such that:
      d(triplets[i, 0],triplets[i, 1]) < d(triplets[i, 0], triplets[i, 2]).

    Returns
    -------
    self : object
      Returns the instance.
    """

    return self._fit(triplets)


class SCML_Supervised(_BaseSCML, TransformerMixin):
  """Supervised version of Sparse Compositional Metric Learning (SCML)

  `SCML_Supervised` creates triplets by taking `k_genuine` neighbours
  of the same class and `k_impostor` neighbours from different classes for each
  point and then runs the SCML algorithm on these triplets.

  Read more in the :ref:`User Guide <scml>`.

  .. warning::
    SCML is still a bit experimental, don't hesitate to report if
    something fails/doesn't work as expected.

  Parameters
  ----------
  beta: float (default=1e-5)
    L1 regularization parameter.

  basis : string or an array-like, optional (default='lda')
    Set of bases to construct the metric. Possible options are
    'lda', and an array-like of shape (n_basis, n_features).

    'lda'
      The `n_basis` basis set is constructed from the LDA of significant
      local regions in the feature space via clustering, for each region
      center k-nearest neighbors are used to obtain the LDA scalings,
      which correspond to the locally discriminative basis.

    array-like
      A matrix of shape (n_basis, n_features), that will be used as
      the basis set for the metric construction.

  n_basis : int, optional
    Number of basis to be yielded. In case it is not set it will be set based
    on `basis`. If no value is selected a default will be computed based on
    the input.

  gamma: float (default = 5e-3)
    Learning rate for the optimization algorithm.

  max_iter : int (default = 100000)
    Number of iterations for the algorithm.

  output_iter : int (default = 5000)
    Number of iterations to check current weights performance and output this
    information in case verbose is True.

  verbose : bool, optional
    If True, prints information while learning.

  preprocessor : array-like, shape=(n_samples, n_features) or callable
    The preprocessor to call to get triplets from indices. If array-like,
    triplets will be formed like this: X[indices].

  random_state : int or numpy.RandomState or None, optional (default=None)
    A pseudo random number generator object or a seed for it if int.

  warm_start : bool, default=False
    When set to True, reuse the solution of the previous call to fit as
    initialization, otherwise, just erase the previous solution.
    Repeatedly calling fit when warm_start is True can result in a different
    solution than when calling fit a single time because of the way the data
    is shuffled.

  Attributes
  ----------
  components_ : `numpy.ndarray`, shape=(n_features, n_features)
    The linear transformation ``L`` deduced from the learned Mahalanobis
    metric (See function `_components_from_basis_weights`.)

  Examples
  --------
  >>> from metric_learn import SCML_Supervised
  >>> from sklearn.datasets import load_iris
  >>> iris_data = load_iris()
  >>> X = iris_data['data']
  >>> Y = iris_data['target']
  >>> scml = SCML_Supervised(random_state=33)
  >>> scml.fit(X, Y)
  SCML_Supervised(random_state=33)
  >>> scml.score_pairs([[X[0], X[1]], [X[0], X[2]]])
  array([1.84640733, 1.55984363])
  >>> scml.get_metric()(X[0], X[1])
  1.8464073327922157

  References
  ----------
  .. [1] Y. Shi, A. Bellet and F. Sha. `Sparse Compositional Metric Learning.
         <http://researchers.lille.inria.fr/abellet/papers/aaai14.pdf>`_. \
         (AAAI), 2014.

  .. [2] Adapted from original `Matlab implementation. \
         <https://github.com/bellet/SCML>`_.

  See Also
  --------
  metric_learn.SCML : The weakly supervised version of this
    algorithm.
  """
  # Add supervised authorized basis construction options
  _authorized_basis = _BaseSCML._authorized_basis + ['lda']

  def __init__(self, k_genuine=3, k_impostor=10, beta=1e-5, basis='lda',
               n_basis=None, gamma=5e-3, max_iter=10000, output_iter=500,
               batch_size=10, verbose=False, preprocessor=None,
               random_state=None, warm_start=False):
    self.k_genuine = k_genuine
    self.k_impostor = k_impostor
    _BaseSCML.__init__(self, beta=beta, basis=basis, n_basis=n_basis,
                       max_iter=max_iter, output_iter=output_iter,
                       batch_size=batch_size, verbose=verbose,
                       preprocessor=preprocessor, random_state=random_state,
                       warm_start=warm_start)

  def fit(self, X, y):
    """Create constraints from labels and learn the SCML model.

    Parameters
    ----------
    X : (n x d) matrix
        Input data, where each row corresponds to a single instance.

    y : (n) array-like
        Data labels.

    Returns
    -------
    self : object
      Returns the instance.
    """
    X, y = self._prepare_inputs(X, y, ensure_min_samples=2)

    basis, n_basis = self._initialize_basis_supervised(X, y)

    if not isinstance(self.k_genuine, int):
      raise ValueError("k_genuine should be an integer, instead it is of type"
                       " %s" % type(self.k_genuine))
    if not isinstance(self.k_impostor, int):
      raise ValueError("k_impostor should be an integer, instead it is of "
                       "type %s" % type(self.k_impostor))

    constraints = Constraints(y)
    triplets = constraints.generate_knntriplets(X, self.k_genuine,
                                                self.k_impostor)

    triplets = X[triplets]

    return self._fit(triplets, basis, n_basis)

  def _initialize_basis_supervised(self, X, y):
    """ Constructs the basis set following one of the supervised options in
    case one is selected.
    """

    if isinstance(self.basis, str) and self.basis == 'lda':
      basis, n_basis = self._generate_bases_LDA(X, y)
    else:
      basis, n_basis = None, None

    return basis, n_basis

  def _generate_bases_LDA(self, X, y):
    """ Generates bases for the 'lda' option.

    The basis set is constructed using Linear Discriminant Analysis of
    significant local regions in the feature space via clustering, for
    each region center k-nearest neighbors are used to obtain the LDA scalings,
    which correspond to the locally discriminative basis. Currently this is
    done at two scales `k={10,20}` if `n_feature < 50` or else `k={20,50}`.
    """

    labels, class_count = np.unique(y, return_counts=True)
    n_class = len(labels)

    n_features = X.shape[1]
    # Number of basis yielded from each LDA
    num_eig = min(n_class-1, n_features)

    if self.n_basis is None:
      # TODO: Get a good default n_basis directive
      n_basis = min(20*n_features, X.shape[0]*2*num_eig - 1)
      warnings.warn('As no value for `n_basis` was selected, the number of '
                    'basis will be set to n_basis= %d' % n_basis)

    elif isinstance(self.n_basis, int):
      n_basis = self.n_basis
    else:
      raise ValueError("n_basis should be an integer, instead it is of type %s"
                       % type(self.n_basis))

    # Number of clusters needed for 2 scales given the number of basis
    # yielded by every LDA
    n_clusters = int(np.ceil(n_basis/(2 * num_eig)))

    if n_basis < n_class:
      warnings.warn("The number of basis is less than the number of classes, "
                    "which may lead to poor discriminative performance.")
    elif n_basis >= X.shape[0]*2*num_eig:
      raise ValueError("Not enough samples to generate %d LDA bases, n_basis"
                       "should be smaller than %d" %
                       (n_basis, X.shape[0]*2*num_eig))

    kmeans = KMeans(n_clusters=n_clusters, n_init=10,
                    random_state=self.random_state, algorithm='elkan').fit(X)
    cX = kmeans.cluster_centers_

    n_scales = 2
    if n_features > 50:
      scales = [20, 50]
    else:
      scales = [10, 20]

    k_class = np.vstack((np.minimum(class_count, scales[0]),
                         np.minimum(class_count, scales[1])))

    idx_set = [np.zeros((n_clusters, sum(k_class[0, :])), dtype=np.int64),
               np.zeros((n_clusters, sum(k_class[1, :])), dtype=np.int64)]

    start_finish_indices = np.hstack((np.zeros((2, 1), np.int64),
                                     k_class)).cumsum(axis=1)

    neigh = NearestNeighbors()

    for c in range(n_class):
        sel_c = np.where(y == labels[c])

        # get k_class same class neighbors
        neigh.fit(X=X[sel_c])
        # Only take the neighbors once for the biggest scale
        neighbors = neigh.kneighbors(X=cX, n_neighbors=k_class[-1, c],
                                     return_distance=False)

        # add index set of neighbors for every cluster center for both scales
        for s, k in enumerate(k_class[:, c]):
          start, finish = start_finish_indices[s, c:c+2]
          idx_set[s][:, start:finish] = np.take(sel_c, neighbors[:, :k])

    # Compute basis for every cluster in both scales
    basis = np.zeros((n_basis, n_features))
    lda = LinearDiscriminantAnalysis()
    start_finish_indices = np.hstack((np.vstack((0, n_clusters * num_eig)),
                                     np.full((2, n_clusters),
                                             num_eig))).cumsum(axis=1)

    for s in range(n_scales):
      for c in range(n_clusters):
        lda.fit(X[idx_set[s][c, :]], y[idx_set[s][c, :]])
        start, finish = start_finish_indices[s, c:c+2]
        normalized_scalings = normalize(lda.scalings_.T)
        try:
          basis[start: finish, :] = normalized_scalings
        except ValueError:
          # handle tail
          basis[start:, :] = normalized_scalings[:n_basis-start]
          break

    return basis, n_basis
