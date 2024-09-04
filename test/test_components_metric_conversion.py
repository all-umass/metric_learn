import unittest
import numpy as np
import pytest
from scipy.stats import ortho_group
from sklearn.datasets import load_iris
from numpy.testing import assert_array_almost_equal, assert_allclose
from metric_learn.sklearn_shims import ignore_warnings

from metric_learn import (
    LMNN_Supervised, NCA, LFDA, Covariance, MLKR,
    LSML_Supervised, ITML_Supervised, SDML_Supervised, RCA_Supervised)
from metric_learn._util import components_from_metric
from metric_learn.exceptions import NonPSDError


class TestTransformerMetricConversion(unittest.TestCase):
  @classmethod
  def setUpClass(self):
    # runs once per test class
    iris_data = load_iris()
    self.X = iris_data['data']
    self.y = iris_data['target']

  def test_cov(self):
    cov = Covariance()
    cov.fit(self.X)
    L = cov.components_
    assert_array_almost_equal(L.T.dot(L), cov.get_mahalanobis_matrix())

  def test_lsml_supervised(self):
    seed = np.random.RandomState(1234)
    lsml = LSML_Supervised(n_constraints=200, random_state=seed)
    lsml.fit(self.X, self.y)
    L = lsml.components_
    assert_array_almost_equal(L.T.dot(L), lsml.get_mahalanobis_matrix())

  def test_itml_supervised(self):
    seed = np.random.RandomState(1234)
    itml = ITML_Supervised(n_constraints=200, random_state=seed)
    itml.fit(self.X, self.y)
    L = itml.components_
    assert_array_almost_equal(L.T.dot(L), itml.get_mahalanobis_matrix())

  def test_lmnn(self):
    lmnn = LMNN_Supervised(n_neighbors=5, learn_rate=1e-6, verbose=False)
    lmnn.fit(self.X, self.y)
    L = lmnn.components_
    assert_array_almost_equal(L.T.dot(L), lmnn.get_mahalanobis_matrix())

  def test_sdml_supervised(self):
    seed = np.random.RandomState(1234)
    sdml = SDML_Supervised(n_constraints=1500, prior='identity',
                           balance_param=1e-5, random_state=seed)
    sdml.fit(self.X, self.y)
    L = sdml.components_
    assert_array_almost_equal(L.T.dot(L), sdml.get_mahalanobis_matrix())

  def test_nca(self):
    n = self.X.shape[0]
    nca = NCA(max_iter=(100000 // n))
    nca.fit(self.X, self.y)
    L = nca.components_
    assert_array_almost_equal(L.T.dot(L), nca.get_mahalanobis_matrix())

  def test_lfda(self):
    lfda = LFDA(k=2, n_components=2)
    lfda.fit(self.X, self.y)
    L = lfda.components_
    assert_array_almost_equal(L.T.dot(L), lfda.get_mahalanobis_matrix())

  def test_rca_supervised(self):
    rca = RCA_Supervised(n_components=2, n_chunks=30, chunk_size=2)
    rca.fit(self.X, self.y)
    L = rca.components_
    assert_array_almost_equal(L.T.dot(L), rca.get_mahalanobis_matrix())

  def test_mlkr(self):
    mlkr = MLKR(n_components=2)
    mlkr.fit(self.X, self.y)
    L = mlkr.components_
    assert_array_almost_equal(L.T.dot(L), mlkr.get_mahalanobis_matrix())

  @ignore_warnings
  def test_components_from_metric_edge_cases(self):
    """Test that components_from_metric returns the right result in various
    edge cases"""
    rng = np.random.RandomState(42)

    # an orthonormal matrix useful for creating matrices with given
    # eigenvalues:
    P = ortho_group.rvs(7, random_state=rng)

    # matrix with all its coefficients very low (to check that the algorithm
    # does not consider it as a diagonal matrix)(non regression test for
    # https://github.com/scikit-learn-contrib/metric-learn/issues/175)
    M = np.diag([1e-15, 2e-16, 3e-15, 4e-16, 5e-15, 6e-16, 7e-15])
    M = P.dot(M).dot(P.T)
    L = components_from_metric(M)
    assert_allclose(L.T.dot(L), M)

    # diagonal matrix
    M = np.diag(np.abs(rng.randn(5)))
    L = components_from_metric(M)
    assert_allclose(L.T.dot(L), M)

    # low-rank matrix (with zeros)
    M = np.zeros((7, 7))
    small_random = rng.randn(3, 3)
    M[:3, :3] = small_random.T.dot(small_random)
    L = components_from_metric(M)
    assert_allclose(L.T.dot(L), M)

    # low-rank matrix (without necessarily zeros)
    R = np.abs(rng.randn(7, 7))
    M = R.dot(np.diag([1, 5, 3, 2, 0, 0, 0])).dot(R.T)
    L = components_from_metric(M)
    assert_allclose(L.T.dot(L), M)

    # matrix with a determinant still high but which is
    # undefinite w.r.t to numpy standards
    M = np.diag([1e5, 1e5, 1e5, 1e5, 1e5, 1e5, 1e-20])
    M = P.dot(M).dot(P.T)
    assert np.abs(np.linalg.det(M)) > 10
    assert np.linalg.slogdet(M)[1] > 1  # (just to show that the computed
    # determinant is far from null)
    assert np.linalg.matrix_rank(M) < M.shape[0]
    # (just to show that this case is indeed considered by numpy as an
    # indefinite case)
    L = components_from_metric(M)
    assert_allclose(L.T.dot(L), M)

    # matrix with lots of small nonzeros that make a big zero when multiplied
    M = np.diag([1e-3, 1e-3, 1e-3, 1e-3, 1e-3, 1e-3, 1e-3])
    L = components_from_metric(M)
    assert_allclose(L.T.dot(L), M)

    # full rank matrix
    M = rng.randn(10, 10)
    M = M.T.dot(M)
    assert np.linalg.matrix_rank(M) == 10
    L = components_from_metric(M)
    assert_allclose(L.T.dot(L), M)

  def test_non_symmetric_matrix_raises(self):
    """Checks that if a non symmetric matrix is given to
    components_from_metric, an error is thrown"""
    rng = np.random.RandomState(42)
    M = rng.randn(10, 10)
    with pytest.raises(ValueError) as raised_error:
      components_from_metric(M)
    assert str(raised_error.value) == "The input metric should be symmetric."

  def test_non_psd_raises(self):
    """Checks that a non PSD matrix (i.e. with negative eigenvalues) will
    raise an error when passed to components_from_metric"""
    rng = np.random.RandomState(42)
    D = np.diag([1, 5, 3, 4.2, -4, -2, 1])
    P = ortho_group.rvs(7, random_state=rng)
    M = P.dot(D).dot(P.T)
    msg = ("Matrix is not positive semidefinite (PSD).")
    with pytest.raises(NonPSDError) as raised_error:
      components_from_metric(M)
    assert str(raised_error.value) == msg
    with pytest.raises(NonPSDError) as raised_error:
      components_from_metric(D)
    assert str(raised_error.value) == msg

  def test_almost_psd_dont_raise(self):
    """Checks that if the metric is almost PSD (i.e. it has some negative
    eigenvalues very close to zero), then components_from_metric will still
    work"""
    rng = np.random.RandomState(42)
    D = np.diag([1, 5, 3, 4.2, -1e-20, -2e-20, -1e-20])
    P = ortho_group.rvs(7, random_state=rng)
    M = P.dot(D).dot(P.T)
    L = components_from_metric(M)
    assert_allclose(L.T.dot(L), M)


if __name__ == '__main__':
  unittest.main()
