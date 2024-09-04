import warnings
import unittest
import re
import pytest
import numpy as np
import scipy
from scipy.optimize import check_grad, approx_fprime
from sklearn.metrics import pairwise_distances, euclidean_distances
from sklearn.datasets import (load_iris, make_classification, make_regression,
                              make_spd_matrix)
from numpy.testing import (assert_array_almost_equal, assert_array_equal,
                           assert_allclose, assert_raises)
from sklearn.exceptions import ConvergenceWarning
from sklearn.utils.validation import check_X_y
from sklearn.preprocessing import StandardScaler
try:
  from inverse_covariance import quic
  assert quic
except ImportError:
  HAS_SKGGM = False
else:
  HAS_SKGGM = True
from metric_learn import (LMNN, NCA, LFDA, Covariance, MLKR, MMC,
                          SCML_Supervised, LSML_Supervised,
                          ITML_Supervised, SDML_Supervised, RCA_Supervised,
                          MMC_Supervised, SDML, RCA, ITML, SCML)
# Import this specially for testing.
from metric_learn.constraints import wrap_pairs, Constraints
from metric_learn.lmnn import _sum_outer_products


def class_separation(X, labels):
  unique_labels, label_inds = np.unique(labels, return_inverse=True)
  ratio = 0
  for li in range(len(unique_labels)):
    Xc = X[label_inds == li]
    Xnc = X[label_inds != li]
    ratio += pairwise_distances(Xc).mean() / pairwise_distances(Xc, Xnc).mean()
  return ratio / len(unique_labels)


class MetricTestCase(unittest.TestCase):
  @classmethod
  def setUpClass(self):
    # runs once per test class
    iris_data = load_iris()
    self.iris_points = iris_data['data']
    self.iris_labels = iris_data['target']
    np.random.seed(1234)


class TestCovariance(MetricTestCase):
  def test_iris(self):
    cov = Covariance()
    cov.fit(self.iris_points)

    csep = class_separation(cov.transform(self.iris_points), self.iris_labels)
    # deterministic result
    self.assertAlmostEqual(csep, 0.72981476)

  def test_singular_returns_pseudo_inverse(self):
    """Checks that if the input covariance matrix is singular, we return
    the pseudo inverse"""
    X, y = load_iris(return_X_y=True)
    # We add a virtual column that is a linear combination of the other
    # columns so that the covariance matrix will be singular
    X = np.concatenate([X, X[:, :2].dot([[2], [3]])], axis=1)
    cov_matrix = np.cov(X, rowvar=False)
    covariance = Covariance()
    covariance.fit(X)
    pseudo_inverse = covariance.get_mahalanobis_matrix()
    # here is the definition of a pseudo inverse according to wikipedia:
    assert_allclose(cov_matrix.dot(pseudo_inverse).dot(cov_matrix),
                    cov_matrix)
    assert_allclose(pseudo_inverse.dot(cov_matrix).dot(pseudo_inverse),
                    pseudo_inverse)


class TestSCML(object):
  @pytest.mark.parametrize('basis', ('lda', 'triplet_diffs'))
  def test_iris(self, basis):
    """
    SCML applied to Iris dataset should give better results when
    computing class separation.
    """
    X, y = load_iris(return_X_y=True)
    before = class_separation(X, y)
    scml = SCML_Supervised(basis=basis, n_basis=85, k_genuine=7, k_impostor=5,
                           random_state=42)
    scml.fit(X, y)
    after = class_separation(scml.transform(X), y)
    assert before > after + 0.03  # It's better by a margin of 0.03

  def test_big_n_features(self):
    X, y = make_classification(n_samples=100, n_classes=3, n_features=60,
                               n_informative=60, n_redundant=0, n_repeated=0,
                               random_state=42)
    X = StandardScaler().fit_transform(X)
    scml = SCML_Supervised(random_state=42, n_basis=399)
    scml.fit(X, y)
    csep = class_separation(scml.transform(X), y)
    assert csep < 0.7

  @pytest.mark.parametrize(('estimator', 'data'),
                           [(SCML, (np.ones((3, 3, 3)),)),
                            (SCML_Supervised, (np.array([[0, 0], [0, 1],
                                                         [2, 0], [2, 1]]),
                                               np.array([1, 0, 1, 0])))])
  def test_bad_basis(self, estimator, data):
    model = estimator(basis='bad_basis', n_basis=33)  # n_basis doesn't matter
    msg = ("`basis` must be one of the options '{}' or an array of shape "
           "(n_basis, n_features)."
           .format("', '".join(model._authorized_basis)))
    with pytest.raises(ValueError) as raised_error:
      model.fit(*data)
    assert msg == raised_error.value.args[0]

  def test_dimension_reduction_msg(self):
    scml = SCML(n_basis=2)
    triplets = np.array([[[0, 1], [2, 1], [0, 0]],
                         [[2, 1], [0, 1], [2, 0]],
                         [[0, 0], [2, 0], [0, 1]],
                         [[2, 0], [0, 0], [2, 1]]])
    msg = ("The number of bases with nonzero weight is less than the "
           "number of features of the input, in consequence the "
           "learned transformation reduces the dimension to 1.")
    with pytest.warns(UserWarning) as raised_warning:
      scml.fit(triplets)
    assert msg == raised_warning[0].message.args[0]

  @pytest.mark.parametrize(('estimator', 'data'),
                           [(SCML, (np.array([[[0, 1], [2, 1], [0, 0]],
                                              [[2, 1], [0, 1], [2, 0]],
                                              [[0, 0], [2, 0], [0, 1]],
                                              [[2, 0], [0, 0], [2, 1]]]),)),
                           (SCML_Supervised, (np.array([[0, 0], [1, 1],
                                                       [3, 3]]),
                                              np.array([1, 2, 3])))])
  def test_n_basis_wrong_type(self, estimator, data):
    n_basis = 4.0
    model = estimator(n_basis=n_basis)
    msg = ("n_basis should be an integer, instead it is of type %s"
           % type(n_basis))
    with pytest.raises(ValueError) as raised_error:
      model.fit(*data)
    assert msg == raised_error.value.args[0]

  def test_small_n_basis_lda(self):
    X = np.array([[0, 0], [1, 1], [2, 2], [3, 3]])
    y = np.array([0, 0, 1, 1])

    n_class = 2
    scml = SCML_Supervised(n_basis=n_class-1)
    msg = ("The number of basis is less than the number of classes, which may"
           " lead to poor discriminative performance.")
    with pytest.warns(UserWarning) as raised_warning:
      scml.fit(X, y)
    assert msg == raised_warning[0].message.args[0]

  def test_big_n_basis_lda(self):
    X = np.array([[0, 0], [1, 1], [3, 3]])
    y = np.array([1, 2, 3])

    n_class = 3
    num_eig = min(n_class - 1, X.shape[1])
    n_basis = X.shape[0] * 2 * num_eig

    scml = SCML_Supervised(n_basis=n_basis)
    msg = ("Not enough samples to generate %d LDA bases, n_basis"
           "should be smaller than %d" %
           (n_basis, n_basis))
    with pytest.raises(ValueError) as raised_error:
      scml.fit(X, y)
    assert msg == raised_error.value.args[0]

  @pytest.mark.parametrize(('estimator', 'data'),
                           [(SCML, (np.random.rand(3, 3, 2),)),
                           (SCML_Supervised, (np.array([[0, 0], [0, 1],
                                                        [2, 0], [2, 1]]),
                                              np.array([1, 0, 1, 0])))])
  def test_array_basis(self, estimator, data):
    """ Test that the proper error is raised when the shape of the input basis
    array is not consistent with the input
    """
    basis = np.eye(3)
    scml = estimator(n_basis=3, basis=basis)

    msg = ('The dimensionality ({}) of the provided bases must match the '
           'dimensionality of the data ({}).'
           .format(basis.shape[1], data[0].shape[-1]))
    with pytest.raises(ValueError) as raised_error:
      scml.fit(*data)
    assert msg == raised_error.value.args[0]

  @pytest.mark.parametrize(('estimator', 'data'),
                           [(SCML, (np.array([[0, 1, 2], [0, 1, 3], [1, 0, 2],
                                              [1, 0, 3], [2, 3, 1], [2, 3, 0],
                                              [3, 2, 1], [3, 2, 0]]),)),
                           (SCML_Supervised, (np.array([0, 1, 2, 3]),
                                              np.array([0, 0, 1, 1])))])
  def test_verbose(self, estimator, data, capsys):
    # assert there is proper output when verbose = True
    model = estimator(preprocessor=np.array([[0, 0], [1, 1], [2, 2], [3, 3]]),
                      max_iter=1, output_iter=1, batch_size=1,
                      basis='triplet_diffs', random_state=42, verbose=True)
    model.fit(*data)
    out, _ = capsys.readouterr()
    expected_out = ('[%s] iter 1\t obj 0.569946\t num_imp 2\n'
                    'max iteration reached.\n' % estimator.__name__)
    assert out == expected_out

  def test_triplet_diffs_toy(self):
    expected_n_basis = 10
    model = SCML_Supervised(n_basis=expected_n_basis)
    X = np.array([[0, 0], [1, 1], [2, 2], [3, 3]])
    triplets = np.array([[0, 1, 2], [0, 1, 3], [1, 0, 2], [1, 0, 3],
                         [2, 3, 1], [2, 3, 0], [3, 2, 1], [3, 2, 0]])
    basis, n_basis = model._generate_bases_dist_diff(triplets, X)
    # All points are along the same line, so the only possible basis will be
    # the vector along that line normalized.
    expected_basis = np.ones((expected_n_basis, 2))/np.sqrt(2)
    assert n_basis == expected_n_basis
    np.testing.assert_allclose(basis, expected_basis)

  def test_lda_toy(self):
    expected_n_basis = 7
    model = SCML_Supervised(n_basis=expected_n_basis)
    X = np.array([[0, 0], [1, 1], [2, 2], [3, 3]])
    y = np.array([0, 0, 1, 1])
    basis, n_basis = model._generate_bases_LDA(X, y)
    # All points are along the same line, so the only possible basis will be
    # the vector along that line normalized. In this case it is possible to
    # obtain it with positive or negative orientations.
    expected_basis = np.ones((expected_n_basis, 2))/np.sqrt(2)
    assert n_basis == expected_n_basis
    np.testing.assert_allclose(np.abs(basis), expected_basis)

  @pytest.mark.parametrize('n_samples', [100, 500])
  @pytest.mark.parametrize('n_features', [10, 50, 100])
  @pytest.mark.parametrize('n_classes', [5, 10, 15])
  def test_triplet_diffs(self, n_samples, n_features, n_classes):
    """
    Test that the correct value of n_basis is being generated with
    different triplet constraints.
    """
    X, y = make_classification(n_samples=n_samples, n_classes=n_classes,
                               n_features=n_features, n_informative=n_features,
                               n_redundant=0, n_repeated=0)
    X = StandardScaler().fit_transform(X)
    model = SCML_Supervised(n_basis=None)  # Explicit n_basis=None
    constraints = Constraints(y)
    triplets = constraints.generate_knntriplets(X, model.k_genuine,
                                                model.k_impostor)

    msg = "As no value for `n_basis` was selected, "
    with pytest.warns(UserWarning) as raised_warning:
      basis, n_basis = model._generate_bases_dist_diff(triplets, X)
    assert msg in str(raised_warning[0].message)

    expected_n_basis = n_features * 80
    assert n_basis == expected_n_basis
    assert basis.shape == (expected_n_basis, n_features)

  @pytest.mark.parametrize('n_samples', [100, 500])
  @pytest.mark.parametrize('n_features', [10, 50, 100])
  @pytest.mark.parametrize('n_classes', [5, 10, 15])
  def test_lda(self, n_samples, n_features, n_classes):
    """
    Test that when n_basis=None, the correct n_basis is generated,
    for SCML_Supervised and different values of n_samples, n_features
    and n_classes.
    """
    X, y = make_classification(n_samples=n_samples, n_classes=n_classes,
                               n_features=n_features, n_informative=n_features,
                               n_redundant=0, n_repeated=0)
    X = StandardScaler().fit_transform(X)

    msg = "As no value for `n_basis` was selected, "
    with pytest.warns(UserWarning) as raised_warning:
      model = SCML_Supervised(n_basis=None)  # Explicit n_basis=None
      basis, n_basis = model._generate_bases_LDA(X, y)
    assert msg in str(raised_warning[0].message)

    num_eig = min(n_classes - 1, n_features)
    expected_n_basis = min(20 * n_features, n_samples * 2 * num_eig - 1)
    assert n_basis == expected_n_basis
    assert basis.shape == (expected_n_basis, n_features)

  @pytest.mark.parametrize('name', ['max_iter', 'output_iter', 'batch_size',
                                    'n_basis'])
  def test_int_inputs(self, name):
    value = 1.0
    d = {name: value}
    scml = SCML(**d)
    triplets = np.array([[[0, 1], [2, 1], [0, 0]]])

    msg = ("%s should be an integer, instead it is of type"
           " %s" % (name, type(value)))
    with pytest.raises(ValueError) as raised_error:
      scml.fit(triplets)
    assert msg == raised_error.value.args[0]

  @pytest.mark.parametrize('name', ['max_iter', 'output_iter', 'batch_size',
                                    'k_genuine', 'k_impostor', 'n_basis'])
  def test_int_inputs_supervised(self, name):
    value = 1.0
    d = {name: value}
    scml = SCML_Supervised(**d)
    X = np.array([[0, 0], [1, 1], [3, 3], [4, 4]])
    y = np.array([1, 1, 0, 0])
    msg = ("%s should be an integer, instead it is of type"
           " %s" % (name, type(value)))
    with pytest.raises(ValueError) as raised_error:
      scml.fit(X, y)
    assert msg == raised_error.value.args[0]

  def test_large_output_iter(self):
    scml = SCML(max_iter=1, output_iter=2, n_basis=33)  # n_basis don't matter
    triplets = np.array([[[0, 1], [2, 1], [0, 0]]])
    msg = ("The value of output_iter must be equal or smaller than"
           " max_iter.")

    with pytest.raises(ValueError) as raised_error:
      scml.fit(triplets)
    assert msg == raised_error.value.args[0]

  @pytest.mark.parametrize("basis", ("lda", "triplet_diffs"))
  def test_warm_start(self, basis):
    X, y = load_iris(return_X_y=True)
    # Test that warm_start=True leads to different weights in each fit call
    scml = SCML_Supervised(basis=basis, n_basis=85, k_genuine=7, k_impostor=5,
                           random_state=42, warm_start=True)
    scml.fit(X, y)
    w_1 = scml.w_
    avg_grad_w_1 = scml.avg_grad_w_
    ada_grad_w_1 = scml.ada_grad_w_
    scml.fit(X, y)
    w_2 = scml.w_
    assert_raises(AssertionError, assert_array_almost_equal, w_1, w_2)
    # And that default warm_start value is False and leads to same
    # weights in each fit call
    scml = SCML_Supervised(basis=basis, n_basis=85, k_genuine=7, k_impostor=5,
                           random_state=42)
    scml.fit(X, y)
    w_3 = scml.w_
    scml.fit(X, y)
    w_4 = scml.w_
    assert_array_almost_equal(w_3, w_4)
    # But would lead to same results with warm_strat=True if same init params
    # were used
    scml.warm_start = True
    scml.w_ = w_1
    scml.avg_grad_w_ = avg_grad_w_1
    scml.ada_grad_w_ = ada_grad_w_1
    scml.fit(X, y)
    w_5 = scml.w_
    assert_array_almost_equal(w_2, w_5)

class TestLSML(MetricTestCase):
  def test_iris(self):
    lsml = LSML_Supervised(n_constraints=200)
    lsml.fit(self.iris_points, self.iris_labels)

    csep = class_separation(lsml.transform(self.iris_points), self.iris_labels)
    self.assertLess(csep, 0.8)  # it's pretty terrible


class TestITML(MetricTestCase):
  def test_iris(self):
    itml = ITML_Supervised(n_constraints=200)
    itml.fit(self.iris_points, self.iris_labels)

    csep = class_separation(itml.transform(self.iris_points), self.iris_labels)
    self.assertLess(csep, 0.2)


@pytest.mark.parametrize('bounds', [None, (20., 100.), [20., 100.],
                                    np.array([20., 100.]),
                                    np.array([[20., 100.]]),
                                    np.array([[20], [100]])])
def test_bounds_parameters_valid(bounds):
  """Asserts that we can provide any array-like of two elements as bounds,
  and that the attribute bound_ is a numpy array"""

  pairs = np.array([[[-10., 0.], [10., 0.]], [[0., 50.], [0., -60]]])
  y_pairs = [1, -1]
  itml = ITML()
  itml.fit(pairs, y_pairs, bounds=bounds)

  X = np.array([[0, 0], [0, 1], [2, 0], [2, 1]])
  y = np.array([1, 0, 1, 0])
  itml_supervised = ITML_Supervised()
  itml_supervised.fit(X, y, bounds=bounds)


@pytest.mark.parametrize('bounds', ['weird', ['weird1', 'weird2'],
                                    np.array([1, 2, 3])])
def test_bounds_parameters_invalid(bounds):
  """Assert that if a non array-like is put for bounds, or an array-like
  of length different than 2, an error is returned"""
  pairs = np.array([[[-10., 0.], [10., 0.]], [[0., 50.], [0., -60]]])
  y_pairs = [1, -1]
  itml = ITML()
  with pytest.raises(Exception):
    itml.fit(pairs, y_pairs, bounds=bounds)

  X = np.array([[0, 0], [0, 1], [2, 0], [2, 1]])
  y = np.array([1, 0, 1, 0])
  itml_supervised = ITML_Supervised()
  with pytest.raises(Exception):
    itml_supervised.fit(X, y, bounds=bounds)


class TestLMNN(MetricTestCase):
  def test_iris(self):
    lmnn = LMNN(n_neighbors=5, learn_rate=1e-6, verbose=False)
    lmnn.fit(self.iris_points, self.iris_labels)

    csep = class_separation(lmnn.transform(self.iris_points),
                            self.iris_labels)
    self.assertLess(csep, 0.25)

  def test_loss_grad_lbfgs(self):
    """Test gradient of loss function
    Assert that the gradient is almost equal to its finite differences
    approximation.
    """
    rng = np.random.RandomState(42)
    X, y = make_classification(random_state=rng)
    L = rng.randn(rng.randint(1, X.shape[1] + 1), X.shape[1])
    lmnn = LMNN()

    k = lmnn.n_neighbors
    reg = lmnn.regularization

    X, y = lmnn._prepare_inputs(X, y, dtype=float,
                                ensure_min_samples=2)
    num_pts, n_components = X.shape
    unique_labels, label_inds = np.unique(y, return_inverse=True)
    lmnn.labels_ = np.arange(len(unique_labels))
    lmnn.components_ = np.eye(n_components)

    target_neighbors = lmnn._select_targets(X, label_inds)

    # sum outer products
    dfG = _sum_outer_products(X, target_neighbors.flatten(),
                              np.repeat(np.arange(X.shape[0]), k))

    # initialize L
    def loss_grad(flat_L):
      return lmnn._loss_grad(X, flat_L.reshape(-1, X.shape[1]), dfG,
                             k, reg, target_neighbors, label_inds)

    def fun(x):
      return loss_grad(x)[1]

    def grad(x):
      return loss_grad(x)[0].ravel()

    # compute relative error
    epsilon = np.sqrt(np.finfo(float).eps)
    rel_diff = (check_grad(fun, grad, L.ravel()) /
                np.linalg.norm(approx_fprime(L.ravel(), fun, epsilon)))
    np.testing.assert_almost_equal(rel_diff, 0., decimal=5)


def test_loss_func(capsys):
  """Test the loss function (and its gradient) on a simple example,
  by comparing the results with the actual implementation of metric-learn,
  with a very simple (but nonperformant) implementation"""

  # toy dataset to use
  X, y = make_classification(n_samples=10, n_classes=2,
                             n_features=6,
                             n_redundant=0, shuffle=True,
                             scale=[1, 1, 20, 20, 20, 20], random_state=42)

  def hinge(a):
    if a > 0:
      return a, 1
    else:
      return 0, 0

  def loss_fn(L, X, y, target_neighbors, reg):
     L = L.reshape(-1, X.shape[1])
     Lx = np.dot(X, L.T)
     loss = 0
     total_active = 0
     grad = np.zeros_like(L)
     for i in range(X.shape[0]):
       for j in target_neighbors[i]:
         loss += (1 - reg) * np.sum((Lx[i] - Lx[j]) ** 2)
         grad += (1 - reg) * np.outer(Lx[i] - Lx[j], X[i] - X[j])
         for k in range(X.shape[0]):
           if y[i] != y[k]:
             hin, active = hinge(1 + np.sum((Lx[i] - Lx[j])**2) -
                                 np.sum((Lx[i] - Lx[k])**2))
             total_active += active
             if active:
               loss += reg * hin
               grad += (reg * (np.outer(Lx[i] - Lx[j], X[i] - X[j]) -
                               np.outer(Lx[i] - Lx[k], X[i] - X[k])))
     grad = 2 * grad
     return grad, loss, total_active

  # we check that the gradient we have computed in the non-performant implem
  # is indeed the true gradient on a toy example:

  def _select_targets(X, y, k):
    target_neighbors = np.empty((X.shape[0], k), dtype=int)
    for label in np.unique(y):
      inds, = np.nonzero(y == label)
      dd = euclidean_distances(X[inds], squared=True)
      np.fill_diagonal(dd, np.inf)
      nn = np.argsort(dd)[..., :k]
      target_neighbors[inds] = inds[nn]
    return target_neighbors

  target_neighbors = _select_targets(X, y, 2)
  regularization = 0.5
  n_features = X.shape[1]
  x0 = np.random.randn(1, n_features)

  def loss(x0):
    return loss_fn(x0.reshape(-1, X.shape[1]), X, y, target_neighbors,
                   regularization)[1]

  def grad(x0):
    return loss_fn(x0.reshape(-1, X.shape[1]), X, y, target_neighbors,
                   regularization)[0].ravel()

  scipy.optimize.check_grad(loss, grad, x0.ravel())

  class LMNN_with_callback(LMNN):
    """ We will use a callback to get the gradient (see later)
    """

    def __init__(self, callback, *args, **kwargs):
      self.callback = callback
      super(LMNN_with_callback, self).__init__(*args, **kwargs)

    def _loss_grad(self, *args, **kwargs):
      grad, objective, total_active = (
          super(LMNN_with_callback, self)._loss_grad(*args, **kwargs))
      self.callback.append(grad)
      return grad, objective, total_active

  class LMNN_nonperformant(LMNN_with_callback):

    def fit(self, X, y):
      self.y = y
      return super(LMNN_nonperformant, self).fit(X, y)

    def _loss_grad(self, X, L, dfG, k, reg, target_neighbors, label_inds):
      grad, loss, total_active = loss_fn(L.ravel(), X, self.y,
                                         target_neighbors, self.regularization)
      self.callback.append(grad)
      return grad, loss, total_active

  mem1, mem2 = [], []
  lmnn_perf = LMNN_with_callback(verbose=True, random_state=42,
                                 init='identity', max_iter=30, callback=mem1)
  lmnn_nonperf = LMNN_nonperformant(verbose=True, random_state=42,
                                    init='identity', max_iter=30,
                                    callback=mem2)
  objectives, obj_diffs, learn_rate, total_active = (dict(), dict(), dict(),
                                                     dict())
  for algo, name in zip([lmnn_perf, lmnn_nonperf], ['perf', 'nonperf']):
    algo.fit(X, y)
    out, _ = capsys.readouterr()
    lines = re.split("\n+", out)
    # we get every variable that is printed from the algorithm in verbose
    num = r'(-?\d+.?\d*(e[+|-]\d+)?)'
    strings = [re.search(r"\d+ (?:{}) (?:{}) (?:(\d+)) (?:{})"
                         .format(num, num, num), s) for s in lines]
    objectives[name] = [float(match.group(1)) for match in strings if match is
                        not None]
    obj_diffs[name] = [float(match.group(3)) for match in strings if match is
                       not None]
    total_active[name] = [float(match.group(5)) for match in strings if
                          match is not
                          None]
    learn_rate[name] = [float(match.group(6)) for match in strings if match is
                        not None]
    assert len(strings) >= 10  # we ensure that we actually did more than 10
    # iterations
    assert total_active[name][0] >= 2  # we ensure that we have some active
    # constraints (that's the case we want to test)
    # we remove the last element because it can be equal to the penultimate
    # if the last gradient update is null
  for i in range(len(mem1)):
    np.testing.assert_allclose(lmnn_perf.callback[i],
                               lmnn_nonperf.callback[i],
                               err_msg='Gradient different at position '
                                       '{}'.format(i))
  np.testing.assert_allclose(objectives['perf'], objectives['nonperf'])
  np.testing.assert_allclose(obj_diffs['perf'], obj_diffs['nonperf'])
  np.testing.assert_allclose(total_active['perf'], total_active['nonperf'])
  np.testing.assert_allclose(learn_rate['perf'], learn_rate['nonperf'])


@pytest.mark.parametrize('X, y, loss', [(np.array([[0], [1], [2], [3]]),
                                         [1, 1, 0, 0], 3.0),
                                        (np.array([[0], [1], [2], [3]]),
                                         [1, 0, 0, 1], 26.)])
def test_toy_ex_lmnn(X, y, loss):
  """Test that the loss give the right result on a toy example"""
  L = np.array([[1]])
  lmnn = LMNN(n_neighbors=1, regularization=0.5)

  k = lmnn.n_neighbors
  reg = lmnn.regularization

  X, y = lmnn._prepare_inputs(X, y, dtype=float,
                              ensure_min_samples=2)
  num_pts, n_components = X.shape
  unique_labels, label_inds = np.unique(y, return_inverse=True)
  lmnn.labels_ = np.arange(len(unique_labels))
  lmnn.components_ = np.eye(n_components)

  target_neighbors = lmnn._select_targets(X, label_inds)

  # sum outer products
  dfG = _sum_outer_products(X, target_neighbors.flatten(),
                            np.repeat(np.arange(X.shape[0]), k))

  # storage
  a1 = [None] * k
  a2 = [None] * k
  for nn_idx in range(k):
    a1[nn_idx] = np.array([])
    a2[nn_idx] = np.array([])

  #  assert that the loss equals the one computed by hand
  assert lmnn._loss_grad(X, L.reshape(-1, X.shape[1]), dfG, k,
                         reg, target_neighbors, label_inds)[1] == loss


def test_convergence_simple_example(capsys):
  # LMNN should converge on this simple example, which it did not with
  # this issue: https://github.com/scikit-learn-contrib/metric-learn/issues/88
  X, y = make_classification(random_state=0)
  lmnn = LMNN(verbose=True)
  lmnn.fit(X, y)
  out, _ = capsys.readouterr()
  assert "LMNN converged with objective" in out


def test_no_twice_same_objective(capsys):
  # test that the objective function never has twice the same value
  # see https://github.com/scikit-learn-contrib/metric-learn/issues/88
  X, y = make_classification(random_state=0)
  lmnn = LMNN(verbose=True)
  lmnn.fit(X, y)
  out, _ = capsys.readouterr()
  lines = re.split("\n+", out)
  # we get only objectives from each line:
  # the regexp matches a float that follows an integer (the iteration
  # number), and which is followed by a (signed) float (delta obj). It
  # matches for instance:
  # 3 **1113.7665747189938** -3.182774197440267 46431.0200999999999998e-06
  objectives = [re.search(r"\d* (?:(\d*.\d*))[ | -]\d*.\d*", s)
                for s in lines]
  objectives = [match.group(1) for match in objectives if match is not None]
  # we remove the last element because it can be equal to the penultimate
  # if the last gradient update is null
  assert len(objectives[:-1]) == len(set(objectives[:-1]))


class TestSDML(MetricTestCase):

  @pytest.mark.skipif(HAS_SKGGM,
                      reason="The warning can be thrown only if skggm is "
                             "not installed.")
  def test_sdml_supervised_raises_warning_msg_not_installed_skggm(self):
    """Tests that the right warning message is raised if someone tries to
    use SDML_Supervised but has not installed skggm, and that the algorithm
    fails to converge"""
    # TODO: remove if we don't need skggm anymore
    # load_iris: dataset where we know scikit-learn's graphical lasso fails
    # with a Floating Point error
    X, y = load_iris(return_X_y=True)
    sdml_supervised = SDML_Supervised(balance_param=0.5, sparsity_param=0.01)
    msg = ("There was a problem in SDML when using scikit-learn's graphical "
           "lasso solver. skggm's graphical lasso can sometimes converge on "
           "non SPD cases where scikit-learn's graphical lasso fails to "
           "converge. Try to install skggm and rerun the algorithm (see "
           "the README.md for the right version of skggm). The following "
           "error message was thrown:")
    with pytest.raises(RuntimeError) as raised_error:
      sdml_supervised.fit(X, y)
    assert str(raised_error.value).startswith(msg)

  @pytest.mark.skipif(HAS_SKGGM,
                      reason="The warning can be thrown only if skggm is "
                             "not installed.")
  def test_sdml_raises_warning_msg_not_installed_skggm(self):
    """Tests that the right warning message is raised if someone tries to
    use SDML but has not installed skggm, and that the algorithm fails to
    converge"""
    # TODO: remove if we don't need skggm anymore
    # case on which we know that scikit-learn's graphical lasso fails
    # because it will return a non SPD matrix
    pairs = np.array([[[-10., 0.], [10., 0.]], [[0., 50.], [0., -60]]])
    y_pairs = [1, -1]
    sdml = SDML(prior='identity', balance_param=100, verbose=True)

    msg = ("There was a problem in SDML when using scikit-learn's graphical "
           "lasso solver. skggm's graphical lasso can sometimes converge on "
           "non SPD cases where scikit-learn's graphical lasso fails to "
           "converge. Try to install skggm and rerun the algorithm (see "
           "the README.md for the right version of skggm).")
    with pytest.raises(RuntimeError) as raised_error:
      sdml.fit(pairs, y_pairs)
    assert msg == str(raised_error.value)

  @pytest.mark.skipif(not HAS_SKGGM,
                      reason="The warning can be thrown only if skggm is "
                             "installed.")
  def test_sdml_raises_warning_msg_installed_skggm(self):
    """Tests that the right warning message is raised if someone tries to
    use SDML and has installed skggm, and that the algorithm fails to
    converge"""
    # TODO: remove if we don't need skggm anymore
    # case on which we know that skggm's graphical lasso fails
    # because it will return non finite values
    pairs = np.array([[[-10., 0.], [10., 0.]], [[0., 50.], [0., -60]]])
    y_pairs = [1, -1]
    sdml = SDML(prior='identity', balance_param=100, verbose=True)

    msg = ("There was a problem in SDML when using skggm's graphical "
           "lasso solver.")
    with pytest.raises(RuntimeError) as raised_error:
      sdml.fit(pairs, y_pairs)
    assert msg == str(raised_error.value)

  @pytest.mark.skipif(not HAS_SKGGM,
                      reason="The warning can be thrown only if skggm is "
                             "installed.")
  def test_sdml_supervised_raises_warning_msg_installed_skggm(self):
    """Tests that the right warning message is raised if someone tries to
    use SDML_Supervised but has not installed skggm, and that the algorithm
    fails to converge"""
    # TODO: remove if we don't need skggm anymore
    # case on which we know that skggm's graphical lasso fails
    # because it will return non finite values
    rng = np.random.RandomState(42)
    # This example will create a diagonal em_cov with a negative coeff (
    # pathological case)
    X = np.array([[-10., 0.], [10., 0.], [5., 0.], [3., 0.]])
    y = [0, 0, 1, 1]
    sdml_supervised = SDML_Supervised(balance_param=0.5, prior='identity',
                                      sparsity_param=0.01, random_state=rng)
    msg = ("There was a problem in SDML when using skggm's graphical "
           "lasso solver.")
    with pytest.raises(RuntimeError) as raised_error:
      sdml_supervised.fit(X, y)
    assert msg == str(raised_error.value)

  @pytest.mark.skipif(not HAS_SKGGM,
                      reason="It's only in the case where skggm is installed"
                             "that no warning should be thrown.")
  def test_raises_no_warning_installed_skggm(self):
    # otherwise we should be able to instantiate and fit SDML and it
    # should raise no error and no ConvergenceWarning
    pairs = np.array([[[-10., 0.], [10., 0.]], [[0., -55.], [0., -60]]])
    y_pairs = [1, -1]
    X, y = make_classification(random_state=42)
    with warnings.catch_warnings(record=True) as records:
      sdml = SDML(prior='covariance')
      sdml.fit(pairs, y_pairs)
    for record in records:
      assert record.category is not ConvergenceWarning
    with warnings.catch_warnings(record=True) as records:
      sdml_supervised = SDML_Supervised(prior='identity', balance_param=1e-5)
      sdml_supervised.fit(X, y)
    for record in records:
      assert record.category is not ConvergenceWarning

  def test_iris(self):
    # Note: this is a flaky test, which fails for certain seeds.
    # TODO: un-flake it!
    rs = np.random.RandomState(5555)

    sdml = SDML_Supervised(n_constraints=1500, prior='identity',
                           balance_param=5e-5, random_state=rs)
    sdml.fit(self.iris_points, self.iris_labels)
    csep = class_separation(sdml.transform(self.iris_points),
                            self.iris_labels)
    self.assertLess(csep, 0.22)

  def test_sdml_raises_warning_non_psd(self):
    """Tests that SDML raises a warning on a toy example where we know the
    pseudo-covariance matrix is not PSD"""
    pairs = np.array([[[-10., 0.], [10., 0.]], [[0., 50.], [0., -60]]])
    y = [1, -1]
    sdml = SDML(prior='covariance', sparsity_param=0.01, balance_param=0.5)
    msg = ("Warning, the input matrix of graphical lasso is not "
           "positive semi-definite (PSD). The algorithm may diverge, "
           "and lead to degenerate solutions. "
           "To prevent that, try to decrease the balance parameter "
           "`balance_param` and/or to set prior='identity'.")
    with pytest.warns(ConvergenceWarning) as raised_warning:
      try:
        sdml.fit(pairs, y)
      except Exception:
        pass
    # we assert that this warning is in one of the warning raised by the
    # estimator
    assert msg in list(map(lambda w: str(w.message), raised_warning))

  def test_sdml_converges_if_psd(self):
    """Tests that sdml converges on a simple problem where we know the
    pseudo-covariance matrix is PSD"""
    pairs = np.array([[[-10., 0.], [10., 0.]], [[0., -55.], [0., -60]]])
    y = [1, -1]
    sdml = SDML(prior='covariance', sparsity_param=0.01, balance_param=0.5)
    sdml.fit(pairs, y)
    assert np.isfinite(sdml.get_mahalanobis_matrix()).all()

  @pytest.mark.skipif(not HAS_SKGGM,
                      reason="sklearn's graphical_lasso can sometimes not "
                             "work on some non SPD problems. We test that "
                             "is works only if skggm is installed.")
  def test_sdml_works_on_non_spd_pb_with_skggm(self):
    """Test that SDML works on a certain non SPD problem on which we know
    it should work, but scikit-learn's graphical_lasso does not work"""
    X, y = load_iris(return_X_y=True)
    sdml = SDML_Supervised(balance_param=0.5, sparsity_param=0.01,
                           prior='covariance',
                           random_state=np.random.RandomState(42))
    sdml.fit(X, y)


@pytest.mark.skipif(not HAS_SKGGM,
                    reason='The message should be printed only if skggm is '
                           'installed.')
def test_verbose_has_installed_skggm_sdml(capsys):
  # Test that if users have installed skggm, a message is printed telling them
  # skggm's solver is used (when they use SDML)
  # TODO: remove if we don't need skggm anymore
  pairs = np.array([[[-10., 0.], [10., 0.]], [[0., -55.], [0., -60]]])
  y_pairs = [1, -1]
  sdml = SDML(verbose=True, prior='covariance')
  sdml.fit(pairs, y_pairs)
  out, _ = capsys.readouterr()
  assert "SDML will use skggm's graphical lasso solver." in out


@pytest.mark.skipif(not HAS_SKGGM,
                    reason='The message should be printed only if skggm is '
                           'installed.')
def test_verbose_has_installed_skggm_sdml_supervised(capsys):
  # Test that if users have installed skggm, a message is printed telling them
  # skggm's solver is used (when they use SDML_Supervised)
  # TODO: remove if we don't need skggm anymore
  X, y = load_iris(return_X_y=True)
  sdml = SDML_Supervised(verbose=True, prior='identity', balance_param=1e-5)
  sdml.fit(X, y)
  out, _ = capsys.readouterr()
  assert "SDML will use skggm's graphical lasso solver." in out


@pytest.mark.skipif(HAS_SKGGM,
                    reason='The message should be printed only if skggm is '
                           'not installed.')
def test_verbose_has_not_installed_skggm_sdml(capsys):
  # Test that if users have installed skggm, a message is printed telling them
  # skggm's solver is used (when they use SDML)
  # TODO: remove if we don't need skggm anymore
  pairs = np.array([[[-10., 0.], [10., 0.]], [[0., -55.], [0., -60]]])
  y_pairs = [1, -1]
  sdml = SDML(verbose=True, prior='covariance')
  sdml.fit(pairs, y_pairs)
  out, _ = capsys.readouterr()
  assert "SDML will use scikit-learn's graphical lasso solver." in out


@pytest.mark.skipif(HAS_SKGGM,
                    reason='The message should be printed only if skggm is '
                           'not installed.')
def test_verbose_has_not_installed_skggm_sdml_supervised(capsys):
  # Test that if users have installed skggm, a message is printed telling them
  # skggm's solver is used (when they use SDML_Supervised)
  # TODO: remove if we don't need skggm anymore
  X, y = make_classification(random_state=42)
  sdml = SDML_Supervised(verbose=True, balance_param=1e-5, prior='identity')
  sdml.fit(X, y)
  out, _ = capsys.readouterr()
  assert "SDML will use scikit-learn's graphical lasso solver." in out


class TestNCA(MetricTestCase):
  def test_iris(self):
    n = self.iris_points.shape[0]

    # Without dimension reduction
    nca = NCA(max_iter=(100000 // n))
    nca.fit(self.iris_points, self.iris_labels)
    csep = class_separation(nca.transform(self.iris_points), self.iris_labels)
    self.assertLess(csep, 0.15)

    # With dimension reduction
    nca = NCA(max_iter=(100000 // n), n_components=2)
    nca.fit(self.iris_points, self.iris_labels)
    csep = class_separation(nca.transform(self.iris_points), self.iris_labels)
    self.assertLess(csep, 0.20)

  def test_finite_differences(self):
    """Test gradient of loss function

    Assert that the gradient is almost equal to its finite differences
    approximation.
    """
    # Initialize the transformation `M`, as well as `X` and `y` and `NCA`
    X, y = make_classification()
    M = np.random.randn(np.random.randint(1, X.shape[1] + 1), X.shape[1])
    mask = y[:, np.newaxis] == y[np.newaxis, :]
    nca = NCA()
    nca.n_iter_ = 0

    def fun(M):
      return nca._loss_grad_lbfgs(M, X, mask)[0]

    def grad(M):
      return nca._loss_grad_lbfgs(M, X, mask)[1].ravel()

    # compute relative error
    epsilon = np.sqrt(np.finfo(float).eps)
    rel_diff = (check_grad(fun, grad, M.ravel()) /
                np.linalg.norm(approx_fprime(M.ravel(), fun, epsilon)))
    np.testing.assert_almost_equal(rel_diff, 0., decimal=6)

  def test_simple_example(self):
    """Test on a simple example.

    Puts four points in the input space where the opposite labels points are
    next to each other. After transform the same labels points should be next
    to each other.

    """
    X = np.array([[0, 0], [0, 1], [2, 0], [2, 1]])
    y = np.array([1, 0, 1, 0])
    nca = NCA(n_components=2,)
    nca.fit(X, y)
    Xansformed = nca.transform(X)
    np.testing.assert_equal(pairwise_distances(Xansformed).argsort()[:, 1],
                            np.array([2, 3, 0, 1]))

  def test_singleton_class(self):
      X = self.iris_points
      y = self.iris_labels

      # one singleton class: test fitting works
      singleton_class = 1
      ind_singleton, = np.where(y == singleton_class)
      y[ind_singleton] = 2
      y[ind_singleton[0]] = singleton_class

      nca = NCA(max_iter=30)
      nca.fit(X, y)

      # One non-singleton class: test fitting works
      ind_1, = np.where(y == 1)
      ind_2, = np.where(y == 2)
      y[ind_1] = 0
      y[ind_1[0]] = 1
      y[ind_2] = 0
      y[ind_2[0]] = 2

      nca = NCA(max_iter=30)
      nca.fit(X, y)

      # Only singleton classes: test fitting does nothing (the gradient
      # must be null in this case, so the final matrix must stay like
      # the initialization)
      ind_0, = np.where(y == 0)
      ind_1, = np.where(y == 1)
      ind_2, = np.where(y == 2)
      X = X[[ind_0[0], ind_1[0], ind_2[0]]]
      y = y[[ind_0[0], ind_1[0], ind_2[0]]]

      A = make_spd_matrix(n_dim=X.shape[1], random_state=X.shape[1])
      nca = NCA(init=A, max_iter=30, n_components=X.shape[1])
      nca.fit(X, y)
      assert_array_equal(nca.components_, A)

  def test_one_class(self):
      # if there is only one class the gradient is null, so the final matrix
      #  must stay like the initialization
      X = self.iris_points[self.iris_labels == 0]
      y = self.iris_labels[self.iris_labels == 0]

      A = make_spd_matrix(n_dim=X.shape[1], random_state=X.shape[1])
      nca = NCA(init=A, max_iter=30, n_components=X.shape[1])
      nca.fit(X, y)
      assert_array_equal(nca.components_, A)


class TestLFDA(MetricTestCase):
  def test_iris(self):
    lfda = LFDA(k=2, n_components=2)
    lfda.fit(self.iris_points, self.iris_labels)
    csep = class_separation(lfda.transform(self.iris_points), self.iris_labels)
    self.assertLess(csep, 0.15)

    # Sanity checks for learned matrices.
    self.assertEqual(lfda.get_mahalanobis_matrix().shape, (4, 4))
    self.assertEqual(lfda.components_.shape, (2, 4))


class TestRCA(MetricTestCase):
  def test_iris(self):
    rca = RCA_Supervised(n_components=2, n_chunks=30, chunk_size=2)
    rca.fit(self.iris_points, self.iris_labels)
    csep = class_separation(rca.transform(self.iris_points), self.iris_labels)
    self.assertLess(csep, 0.29)

  def test_rank_deficient_returns_warning(self):
    """Checks that if the covariance matrix is not invertible, we raise a
    warning message advising to use PCA"""
    X, y = load_iris(return_X_y=True)
    # we make the fourth column a linear combination of the two first,
    # so that the covariance matrix will not be invertible:
    X[:, 3] = X[:, 0] + 3 * X[:, 1]
    rca = RCA()
    msg = ('The inner covariance matrix is not invertible, '
           'so the transformation matrix may contain Nan values. '
           'You should remove any linearly dependent features and/or '
           'reduce the dimensionality of your input, '
           'for instance using `sklearn.decomposition.PCA` as a '
           'preprocessing step.')

    with warnings.catch_warnings(record=True) as raised_warnings:
      rca.fit(X, y)
    assert any(str(w.message) == msg for w in raised_warnings)

  def test_unknown_labels(self):
    n = 200
    n_chunks = 50
    X, y = make_classification(random_state=42, n_samples=2 * n,
                               n_features=6, n_informative=6, n_redundant=0)
    y2 = np.concatenate((y[:n], -np.ones(n)))

    rca = RCA_Supervised(n_chunks=n_chunks, random_state=42)
    rca.fit(X[:n], y[:n])

    rca2 = RCA_Supervised(n_chunks=n_chunks, random_state=42)
    rca2.fit(X, y2)

    assert not np.any(np.isnan(rca.components_))
    assert not np.any(np.isnan(rca2.components_))

    np.testing.assert_array_equal(rca.components_, rca2.components_)

  def test_bad_parameters(self):
    n = 200
    n_chunks = 3
    X, y = make_classification(random_state=42, n_samples=n,
                               n_features=6, n_informative=6, n_redundant=0)

    rca = RCA_Supervised(n_chunks=n_chunks, random_state=42)
    msg = ('Due to the parameters of RCA_Supervised, '
           'the inner covariance matrix is not invertible, '
           'so the transformation matrix will contain Nan values. '
           'Increase the number or size of the chunks to correct '
           'this problem.'
           )
    with warnings.catch_warnings(record=True) as raised_warning:
      rca.fit(X, y)
    assert any(str(w.message) == msg for w in raised_warning)


class TestMLKR(MetricTestCase):
  def test_iris(self):
    mlkr = MLKR()
    mlkr.fit(self.iris_points, self.iris_labels)
    csep = class_separation(mlkr.transform(self.iris_points), self.iris_labels)
    self.assertLess(csep, 0.25)

  def test_finite_differences(self):
    """Test gradient of loss function

    Assert that the gradient is almost equal to its finite differences
    approximation.
    """
    # Initialize the transformation `M`, as well as `X`, and `y` and `MLKR`
    X, y = make_regression(n_features=4, random_state=1, n_samples=20)
    X, y = check_X_y(X, y)
    M = np.random.randn(2, X.shape[1])
    mlkr = MLKR()
    mlkr.n_iter_ = 0

    def fun(M):
      return mlkr._loss(M, X, y)[0]

    def grad_fn(M):
      return mlkr._loss(M, X, y)[1].ravel()

    # compute relative error
    rel_diff = check_grad(fun, grad_fn, M.ravel()) / np.linalg.norm(grad_fn(M))
    np.testing.assert_almost_equal(rel_diff, 0.)


class TestMMC(MetricTestCase):
  def test_iris(self):

    # Generate full set of constraints for comparison with reference
    # implementation
    mask = self.iris_labels[None] == self.iris_labels[:, None]
    a, b = np.nonzero(np.triu(mask, k=1))
    c, d = np.nonzero(np.triu(~mask, k=1))

    # Full metric
    n_features = self.iris_points.shape[1]
    mmc = MMC(tol=0.01, init=np.eye(n_features) / 10)
    mmc.fit(*wrap_pairs(self.iris_points, [a, b, c, d]))
    expected = [[+0.000514, +0.000868, -0.001195, -0.001703],
                [+0.000868, +0.001468, -0.002021, -0.002879],
                [-0.001195, -0.002021, +0.002782, +0.003964],
                [-0.001703, -0.002879, +0.003964, +0.005648]]
    assert_array_almost_equal(expected, mmc.get_mahalanobis_matrix(),
                              decimal=6)

    # Diagonal metric
    mmc = MMC(diagonal=True)
    mmc.fit(*wrap_pairs(self.iris_points, [a, b, c, d]))
    expected = [0, 0, 1.210220, 1.228596]
    assert_array_almost_equal(np.diag(expected), mmc.get_mahalanobis_matrix(),
                              decimal=6)

    # Supervised Full
    mmc = MMC_Supervised()
    mmc.fit(self.iris_points, self.iris_labels)
    csep = class_separation(mmc.transform(self.iris_points), self.iris_labels)
    self.assertLess(csep, 0.15)

    # Supervised Diagonal
    mmc = MMC_Supervised(diagonal=True)
    mmc.fit(self.iris_points, self.iris_labels)
    csep = class_separation(mmc.transform(self.iris_points), self.iris_labels)
    self.assertLess(csep, 0.2)


@pytest.mark.parametrize(('algo_class', 'dataset'),
                         [(NCA, make_classification()),
                          (MLKR, make_regression())])
def test_verbose(algo_class, dataset, capsys):
  # assert there is proper output when verbose = True
  X, y = dataset
  model = algo_class(verbose=True)
  model.fit(X, y)
  out, _ = capsys.readouterr()

  # check output
  lines = re.split('\n+', out)
  header = '{:>10} {:>20} {:>10}'.format('Iteration', 'Objective Value',
                                         'Time(s)')
  assert lines[0] == '[{}]'.format(algo_class.__name__)
  assert lines[1] == '[{}] {}'.format(algo_class.__name__, header)
  assert lines[2] == '[{}] {}'.format(algo_class.__name__, '-' * len(header))
  for line in lines[3:-2]:
    # The following regex will match for instance:
    # '[NCA]          0         6.988936e+01       0.01'
    assert re.match(r"\[" + algo_class.__name__ + r"\]\ *\d+\ *\d\.\d{6}e[+|-]"
                    r"\d+\ *\d+\.\d{2}", line)
  assert re.match(r"\[" + algo_class.__name__ + r"\] Training took\ *"
                  r"\d+\.\d{2}s\.", lines[-2])
  assert lines[-1] == ''


@pytest.mark.parametrize(('algo_class', 'dataset'),
                         [(NCA, make_classification()),
                          (MLKR, make_regression(n_features=10))])
def test_no_verbose(dataset, algo_class, capsys):
  # assert by default there is no output (verbose=False)
  X, y = dataset
  model = algo_class()
  model.fit(X, y)
  out, _ = capsys.readouterr()
  # check output
  assert (out == '')


@pytest.mark.parametrize(('algo_class', 'dataset'),
                         [(NCA, make_classification()),
                          (MLKR, make_regression(n_features=10))])
def test_convergence_warning(dataset, algo_class):
    X, y = dataset
    model = algo_class(max_iter=2, verbose=True)
    cls_name = model.__class__.__name__
    msg = '[{}] {} did not converge'.format(cls_name, cls_name)
    with pytest.warns(Warning) as raised_warning:
      model.fit(X, y)
    assert any([msg in str(warn.message) for warn in raised_warning])


if __name__ == '__main__':
  unittest.main()
