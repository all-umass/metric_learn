import unittest
import numpy as np
from sklearn.datasets import load_iris
from numpy.testing import assert_array_almost_equal

from metric_learn import (
    LMNN_Supervised, NCA, LFDA, Covariance, MLKR,
    LSML_Supervised, ITML_Supervised, SDML_Supervised, RCA_Supervised,
    MMC_Supervised)


class TestFitTransform(unittest.TestCase):
  @classmethod
  def setUpClass(self):
    # runs once per test class
    iris_data = load_iris()
    self.X = iris_data['data']
    self.y = iris_data['target']

  def test_cov(self):
    cov = Covariance()
    cov.fit(self.X)
    res_1 = cov.transform(self.X)

    cov = Covariance()
    res_2 = cov.fit_transform(self.X)
    # deterministic result
    assert_array_almost_equal(res_1, res_2)

  def test_lsml_supervised(self):
    seed = np.random.RandomState(1234)
    lsml = LSML_Supervised(n_constraints=200, random_state=seed)
    lsml.fit(self.X, self.y)
    res_1 = lsml.transform(self.X)

    seed = np.random.RandomState(1234)
    lsml = LSML_Supervised(n_constraints=200, random_state=seed)
    res_2 = lsml.fit_transform(self.X, self.y)

    assert_array_almost_equal(res_1, res_2)

  def test_itml_supervised(self):
    seed = np.random.RandomState(1234)
    itml = ITML_Supervised(n_constraints=200, random_state=seed)
    itml.fit(self.X, self.y)
    res_1 = itml.transform(self.X)

    seed = np.random.RandomState(1234)
    itml = ITML_Supervised(n_constraints=200, random_state=seed)
    res_2 = itml.fit_transform(self.X, self.y)

    assert_array_almost_equal(res_1, res_2)

  def test_lmnn(self):
    lmnn = LMNN_Supervised(n_neighbors=5, learn_rate=1e-6, verbose=False)
    lmnn.fit(self.X, self.y)
    res_1 = lmnn.transform(self.X)

    lmnn = LMNN_Supervised(n_neighbors=5, learn_rate=1e-6, verbose=False)
    res_2 = lmnn.fit_transform(self.X, self.y)

    assert_array_almost_equal(res_1, res_2)

  def test_sdml_supervised(self):
    seed = np.random.RandomState(1234)
    sdml = SDML_Supervised(n_constraints=1500, balance_param=1e-5,
                           prior='identity', random_state=seed)
    sdml.fit(self.X, self.y)
    res_1 = sdml.transform(self.X)

    seed = np.random.RandomState(1234)
    sdml = SDML_Supervised(n_constraints=1500, balance_param=1e-5,
                           prior='identity', random_state=seed)
    res_2 = sdml.fit_transform(self.X, self.y)

    assert_array_almost_equal(res_1, res_2)

  def test_nca(self):
    n = self.X.shape[0]
    nca = NCA(max_iter=(100000 // n))
    nca.fit(self.X, self.y)
    res_1 = nca.transform(self.X)

    nca = NCA(max_iter=(100000 // n))
    res_2 = nca.fit_transform(self.X, self.y)

    assert_array_almost_equal(res_1, res_2)

  def test_lfda(self):
    lfda = LFDA(k=2, n_components=2)
    lfda.fit(self.X, self.y)
    res_1 = lfda.transform(self.X)

    lfda = LFDA(k=2, n_components=2)
    res_2 = lfda.fit_transform(self.X, self.y)

    # signs may be flipped, that's okay
    assert_array_almost_equal(abs(res_1), abs(res_2))

  def test_rca_supervised(self):
    seed = np.random.RandomState(1234)
    rca = RCA_Supervised(n_components=2, n_chunks=30, chunk_size=2,
                         random_state=seed)
    rca.fit(self.X, self.y)
    res_1 = rca.transform(self.X)

    seed = np.random.RandomState(1234)
    rca = RCA_Supervised(n_components=2, n_chunks=30, chunk_size=2,
                         random_state=seed)
    res_2 = rca.fit_transform(self.X, self.y)

    assert_array_almost_equal(res_1, res_2)

  def test_mlkr(self):
    mlkr = MLKR(n_components=2)
    mlkr.fit(self.X, self.y)
    res_1 = mlkr.transform(self.X)

    mlkr = MLKR(n_components=2)
    res_2 = mlkr.fit_transform(self.X, self.y)

    assert_array_almost_equal(res_1, res_2)

  def test_mmc_supervised(self):
    seed = np.random.RandomState(1234)
    mmc = MMC_Supervised(n_constraints=200, random_state=seed)
    mmc.fit(self.X, self.y)
    res_1 = mmc.transform(self.X)

    seed = np.random.RandomState(1234)
    mmc = MMC_Supervised(n_constraints=200, random_state=seed)
    res_2 = mmc.fit_transform(self.X, self.y)

    assert_array_almost_equal(res_1, res_2)


if __name__ == '__main__':
  unittest.main()
