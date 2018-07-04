import unittest
import numpy as np
from scipy.optimize import check_grad
from six.moves import xrange
from sklearn.metrics import pairwise_distances
from sklearn.datasets import load_iris, make_classification
from numpy.testing import assert_array_almost_equal, assert_array_equal
from sklearn.utils.testing import assert_warns_message

from metric_learn import (LMNN, NCA, LFDA, Covariance, MLKR, MMC,
                          LSML_Supervised, ITML_Supervised, SDML_Supervised,
                          RCA_Supervised, MMC_Supervised)
from metric_learn.lmnn import python_LMNN


def class_separation(X, labels):
  unique_labels, label_inds = np.unique(labels, return_inverse=True)
  ratio = 0
  for li in xrange(len(unique_labels)):
    Xc = X[label_inds==li]
    Xnc = X[label_inds!=li]
    ratio += pairwise_distances(Xc).mean() / pairwise_distances(Xc,Xnc).mean()
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

    csep = class_separation(cov.transform(), self.iris_labels)
    # deterministic result
    self.assertAlmostEqual(csep, 0.73068122)


class TestLSML(MetricTestCase):
  def test_iris(self):
    lsml = LSML_Supervised(num_constraints=200)
    lsml.fit(self.iris_points, self.iris_labels)

    csep = class_separation(lsml.transform(), self.iris_labels)
    self.assertLess(csep, 0.8)  # it's pretty terrible


class TestITML(MetricTestCase):
  def test_iris(self):
    itml = ITML_Supervised(num_constraints=200)
    itml.fit(self.iris_points, self.iris_labels)

    csep = class_separation(itml.transform(), self.iris_labels)
    self.assertLess(csep, 0.2)


class TestLMNN(MetricTestCase):
  def test_iris(self):
    # Test both impls, if available.
    for LMNN_cls in set((LMNN, python_LMNN)):
      lmnn = LMNN_cls(k=5, learn_rate=1e-6, verbose=False)
      lmnn.fit(self.iris_points, self.iris_labels)

      csep = class_separation(lmnn.transform(), self.iris_labels)
      self.assertLess(csep, 0.25)


class TestSDML(MetricTestCase):
  def test_iris(self):
    # Note: this is a flaky test, which fails for certain seeds.
    # TODO: un-flake it!
    rs = np.random.RandomState(5555)

    sdml = SDML_Supervised(num_constraints=1500)
    sdml.fit(self.iris_points, self.iris_labels, random_state=rs)
    csep = class_separation(sdml.transform(), self.iris_labels)
    self.assertLess(csep, 0.25)


class TestNCA(MetricTestCase):
  def test_iris(self):
    n = self.iris_points.shape[0]

    # Without dimension reduction
    nca = NCA(max_iter=(100000//n))
    nca.fit(self.iris_points, self.iris_labels)
    csep = class_separation(nca.transform(), self.iris_labels)
    self.assertLess(csep, 0.15)

    # With dimension reduction
    nca = NCA(max_iter=(100000//n), num_dims=2, tol=1e-9)
    nca.fit(self.iris_points, self.iris_labels)
    csep = class_separation(nca.transform(), self.iris_labels)
    self.assertLess(csep, 0.15)

  def test_finite_differences(self):
    """Test gradient of loss function

    Assert that the gradient is almost equal to its finite differences
    approximation.
    """
    # Initialize the transformation `M`, as well as `X` and `y` and `NCA`
    X, y = make_classification()
    M = np.random.randn(np.random.randint(1, X.shape[1] + 1), X.shape[1])
    mask = y[:, np.newaxis] == y[np.newaxis, :]

    def fun(M):
      return NCA._loss_grad_lbfgs(M, X, mask)[0]

    def grad(M):
      return NCA._loss_grad_lbfgs(M, X, mask)[1].ravel()

    # compute relative error
    rel_diff = check_grad(fun, grad, M.ravel()) / np.linalg.norm(grad(M))
    np.testing.assert_almost_equal(rel_diff, 0., decimal=6)

  def test_simple_example(self):
    """Test on a simple example.

    Puts four points in the input space where the opposite labels points are
    next to each other. After transform the same labels points should be next
    to each other.

    """
    X = np.array([[0, 0], [0, 1], [2, 0], [2, 1]])
    y = np.array([1, 0, 1, 0])
    nca = NCA(num_dims=2,)
    nca.fit(X, y)
    Xansformed = nca.transform(X)
    np.testing.assert_equal(pairwise_distances(Xansformed).argsort()[:, 1],
                            np.array([2, 3, 0, 1]))

  def test_deprecation(self):
    # test that the right deprecation message is thrown.
    # TODO: remove in v.0.5
    X = np.array([[0, 0], [0, 1], [2, 0], [2, 1]])
    y = np.array([1, 0, 1, 0])
    nca = NCA(num_dims=2, learning_rate=0.01)
    msg = ('"learning_rate" parameter is not used.'
           ' It has been deprecated in version 0.4 and will be'
           'removed in 0.5')
    assert_warns_message(DeprecationWarning, msg, nca.fit, X, y)

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

      EPS = np.finfo(float).eps
      A = np.zeros((X.shape[1], X.shape[1]))
      np.fill_diagonal(A,
                       1. / (np.maximum(X.max(axis=0) - X.min(axis=0), EPS)))
      nca = NCA(max_iter=30, num_dims=X.shape[1])
      nca.fit(X, y)
      assert_array_equal(nca.A_, A)

  def test_one_class(self):
      # if there is only one class the gradient is null, so the final matrix
      #  must stay like the initialization
      X = self.iris_points[self.iris_labels == 0]
      y = self.iris_labels[self.iris_labels == 0]
      EPS = np.finfo(float).eps
      A = np.zeros((X.shape[1], X.shape[1]))
      np.fill_diagonal(A,
                       1. / (np.maximum(X.max(axis=0) - X.min(axis=0), EPS)))
      nca = NCA(max_iter=30, num_dims=X.shape[1])
      nca.fit(X, y)
      assert_array_equal(nca.A_, A)


class TestLFDA(MetricTestCase):
  def test_iris(self):
    lfda = LFDA(k=2, num_dims=2)
    lfda.fit(self.iris_points, self.iris_labels)
    csep = class_separation(lfda.transform(), self.iris_labels)
    self.assertLess(csep, 0.15)

    # Sanity checks for learned matrices.
    self.assertEqual(lfda.metric().shape, (4, 4))
    self.assertEqual(lfda.transformer().shape, (2, 4))


class TestRCA(MetricTestCase):
  def test_iris(self):
    rca = RCA_Supervised(num_dims=2, num_chunks=30, chunk_size=2)
    rca.fit(self.iris_points, self.iris_labels)
    csep = class_separation(rca.transform(), self.iris_labels)
    self.assertLess(csep, 0.25)

  def test_feature_null_variance(self):
    X = np.hstack((self.iris_points, np.eye(len(self.iris_points), M=1)))

    # Apply PCA with the number of components
    rca = RCA_Supervised(num_dims=2, pca_comps=3, num_chunks=30, chunk_size=2)
    rca.fit(X, self.iris_labels)
    csep = class_separation(rca.transform(), self.iris_labels)
    self.assertLess(csep, 0.30)

    # Apply PCA with the minimum variance ratio
    rca = RCA_Supervised(num_dims=2, pca_comps=0.95, num_chunks=30,
                         chunk_size=2)
    rca.fit(X, self.iris_labels)
    csep = class_separation(rca.transform(), self.iris_labels)
    self.assertLess(csep, 0.30)


class TestMLKR(MetricTestCase):
  def test_iris(self):
    mlkr = MLKR()
    mlkr.fit(self.iris_points, self.iris_labels)
    csep = class_separation(mlkr.transform(), self.iris_labels)
    self.assertLess(csep, 0.25)


class TestMMC(MetricTestCase):
  def test_iris(self):

    # Generate full set of constraints for comparison with reference implementation
    n = self.iris_points.shape[0]
    mask = (self.iris_labels[None] == self.iris_labels[:,None])
    a, b = np.nonzero(np.triu(mask, k=1))
    c, d = np.nonzero(np.triu(~mask, k=1))

    # Full metric
    mmc = MMC(convergence_threshold=0.01)
    mmc.fit(self.iris_points, [a,b,c,d])
    expected = [[+0.00046504, +0.00083371, -0.00111959, -0.00165265],
                [+0.00083371, +0.00149466, -0.00200719, -0.00296284],
                [-0.00111959, -0.00200719, +0.00269546, +0.00397881],
                [-0.00165265, -0.00296284, +0.00397881, +0.00587320]]
    assert_array_almost_equal(expected, mmc.metric(), decimal=6)

    # Diagonal metric
    mmc = MMC(diagonal=True)
    mmc.fit(self.iris_points, [a,b,c,d])
    expected = [0, 0, 1.21045968, 1.22552608]
    assert_array_almost_equal(np.diag(expected), mmc.metric(), decimal=6)
    
    # Supervised Full
    mmc = MMC_Supervised()
    mmc.fit(self.iris_points, self.iris_labels)
    csep = class_separation(mmc.transform(), self.iris_labels)
    self.assertLess(csep, 0.15)
    
    # Supervised Diagonal
    mmc = MMC_Supervised(diagonal=True)
    mmc.fit(self.iris_points, self.iris_labels)
    csep = class_separation(mmc.transform(), self.iris_labels)
    self.assertLess(csep, 0.2)


if __name__ == '__main__':
  unittest.main()
