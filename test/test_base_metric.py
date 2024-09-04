from numpy.core.numeric import array_equal
import warnings
import pytest
import re
import unittest
import metric_learn
import numpy as np
from sklearn import clone
from test.test_utils import ids_metric_learners, metric_learners, remove_y
from metric_learn.sklearn_shims import set_random_state, SKLEARN_AT_LEAST_0_22


def remove_spaces(s):
  return re.sub(r'\s+', '', s)


def sk_repr_kwargs(def_kwargs, nndef_kwargs):
    """Given the non-default arguments, and the default
    keywords arguments, build the string that will appear
    in the __repr__ of the estimator, depending on the
    version of scikit-learn.
    """
    if SKLEARN_AT_LEAST_0_22:
        def_kwargs = {}
    def_kwargs.update(nndef_kwargs)
    args_str = ",".join(f"{key}={repr(value)}"
                        for key, value in def_kwargs.items())
    return args_str


class TestStringRepr(unittest.TestCase):

  def test_covariance(self):
    def_kwargs = {'preprocessor': None}
    nndef_kwargs = {}
    merged_kwargs = sk_repr_kwargs(def_kwargs, nndef_kwargs)
    self.assertEqual(remove_spaces(str(metric_learn.Covariance())),
                     remove_spaces(f"Covariance({merged_kwargs})"))

  def test_lmnn(self):
    def_kwargs = {'convergence_tol': 0.001, 'init': 'auto', 'n_neighbors': 3,
                  'learn_rate': 1e-07, 'max_iter': 1000, 'min_iter': 50,
                  'n_components': None, 'preprocessor': None,
                  'random_state': None, 'regularization': 0.5,
                  'verbose': False}
    nndef_kwargs = {'convergence_tol': 0.01, 'n_neighbors': 6}
    merged_kwargs = sk_repr_kwargs(def_kwargs, nndef_kwargs)
    self.assertEqual(
        remove_spaces(str(metric_learn.LMNN_Supervised(convergence_tol=0.01,
                                            n_neighbors=6))),
        remove_spaces(f"LMNN_Supervised({merged_kwargs})"))

  def test_nca(self):
    def_kwargs = {'init': 'auto', 'max_iter': 100, 'n_components': None,
                  'preprocessor': None, 'random_state': None, 'tol': None,
                  'verbose': False}
    nndef_kwargs = {'max_iter': 42}
    merged_kwargs = sk_repr_kwargs(def_kwargs, nndef_kwargs)
    self.assertEqual(remove_spaces(str(metric_learn.NCA(max_iter=42))),
                     remove_spaces(f"NCA({merged_kwargs})"))

  def test_lfda(self):
    def_kwargs = {'embedding_type': 'weighted', 'k': None,
                  'n_components': None, 'preprocessor': None}
    nndef_kwargs = {'k': 2}
    merged_kwargs = sk_repr_kwargs(def_kwargs, nndef_kwargs)
    self.assertEqual(remove_spaces(str(metric_learn.LFDA(k=2))),
                     remove_spaces(f"LFDA({merged_kwargs})"))

  def test_itml(self):
    def_kwargs = {'tol': 0.001, 'gamma': 1.0,
                  'max_iter': 1000, 'preprocessor': None,
                  'prior': 'identity', 'random_state': None, 'verbose': False}
    nndef_kwargs = {'gamma': 0.5}
    merged_kwargs = sk_repr_kwargs(def_kwargs, nndef_kwargs)
    self.assertEqual(remove_spaces(str(metric_learn.ITML(gamma=0.5))),
                     remove_spaces(f"ITML({merged_kwargs})"))
    def_kwargs = {'tol': 0.001, 'gamma': 1.0,
                  'max_iter': 1000, 'n_constraints': None,
                  'preprocessor': None, 'prior': 'identity',
                  'random_state': None, 'verbose': False}
    nndef_kwargs = {'n_constraints': 7}
    merged_kwargs = sk_repr_kwargs(def_kwargs, nndef_kwargs)
    self.assertEqual(
        remove_spaces(str(metric_learn.ITML_Supervised(n_constraints=7))),
        remove_spaces(f"ITML_Supervised({merged_kwargs})"))

  def test_lsml(self):
    def_kwargs = {'max_iter': 1000, 'preprocessor': None, 'prior': 'identity',
                  'random_state': None, 'tol': 0.001, 'verbose': False}
    nndef_kwargs = {'tol': 0.1}
    merged_kwargs = sk_repr_kwargs(def_kwargs, nndef_kwargs)
    self.assertEqual(remove_spaces(str(metric_learn.LSML(tol=0.1))),
                     remove_spaces(f"LSML({merged_kwargs})"))
    def_kwargs = {'max_iter': 1000, 'n_constraints': None,
                  'preprocessor': None, 'prior': 'identity',
                  'random_state': None, 'tol': 0.001, 'verbose': False,
                  'weights': None}
    nndef_kwargs = {'verbose': True}
    merged_kwargs = sk_repr_kwargs(def_kwargs, nndef_kwargs)
    self.assertEqual(
        remove_spaces(str(metric_learn.LSML_Supervised(verbose=True))),
        remove_spaces(f"LSML_Supervised({merged_kwargs})"))

  def test_sdml(self):
    def_kwargs = {'balance_param': 0.5, 'preprocessor': None,
                  'prior': 'identity', 'random_state': None,
                  'sparsity_param': 0.01, 'verbose': False}
    nndef_kwargs = {'verbose': True}
    merged_kwargs = sk_repr_kwargs(def_kwargs, nndef_kwargs)
    self.assertEqual(remove_spaces(str(metric_learn.SDML(verbose=True))),
                     remove_spaces(f"SDML({merged_kwargs})"))
    def_kwargs = {'balance_param': 0.5, 'n_constraints': None,
                  'preprocessor': None, 'prior': 'identity',
                  'random_state': None, 'sparsity_param': 0.01,
                  'verbose': False}
    nndef_kwargs = {'sparsity_param': 0.5}
    merged_kwargs = sk_repr_kwargs(def_kwargs, nndef_kwargs)
    self.assertEqual(
        remove_spaces(str(metric_learn.SDML_Supervised(sparsity_param=0.5))),
        remove_spaces(f"SDML_Supervised({merged_kwargs})"))

  def test_rca(self):
    def_kwargs = {'n_components': None, 'preprocessor': None}
    nndef_kwargs = {'n_components': 3}
    merged_kwargs = sk_repr_kwargs(def_kwargs, nndef_kwargs)
    self.assertEqual(remove_spaces(str(metric_learn.RCA(n_components=3))),
                     remove_spaces(f"RCA({merged_kwargs})"))
    def_kwargs = {'chunk_size': 2, 'n_components': None, 'n_chunks': 100,
                  'preprocessor': None, 'random_state': None}
    nndef_kwargs = {'n_chunks': 5}
    merged_kwargs = sk_repr_kwargs(def_kwargs, nndef_kwargs)
    self.assertEqual(
        remove_spaces(str(metric_learn.RCA_Supervised(n_chunks=5))),
        remove_spaces(f"RCA_Supervised({merged_kwargs})"))

  def test_mlkr(self):
    def_kwargs = {'init': 'auto', 'max_iter': 1000,
                  'n_components': None, 'preprocessor': None,
                  'random_state': None, 'tol': None, 'verbose': False}
    nndef_kwargs = {'max_iter': 777}
    merged_kwargs = sk_repr_kwargs(def_kwargs, nndef_kwargs)
    self.assertEqual(remove_spaces(str(metric_learn.MLKR(max_iter=777))),
                     remove_spaces(f"MLKR({merged_kwargs})"))

  def test_mmc(self):
    def_kwargs = {'tol': 0.001, 'diagonal': False,
                  'diagonal_c': 1.0, 'init': 'identity', 'max_iter': 100,
                  'max_proj': 10000, 'preprocessor': None,
                  'random_state': None, 'verbose': False}
    nndef_kwargs = {'diagonal': True}
    merged_kwargs = sk_repr_kwargs(def_kwargs, nndef_kwargs)
    self.assertEqual(remove_spaces(str(metric_learn.MMC(diagonal=True))),
                     remove_spaces(f"MMC({merged_kwargs})"))
    def_kwargs = {'tol': 1e-06, 'diagonal': False,
                  'diagonal_c': 1.0, 'init': 'identity', 'max_iter': 100,
                  'max_proj': 10000, 'n_constraints': None,
                  'preprocessor': None, 'random_state': None,
                  'verbose': False}
    nndef_kwargs = {'max_iter': 1}
    merged_kwargs = sk_repr_kwargs(def_kwargs, nndef_kwargs)
    self.assertEqual(
        remove_spaces(str(metric_learn.MMC_Supervised(max_iter=1))),
        remove_spaces(f"MMC_Supervised({merged_kwargs})"))


@pytest.mark.parametrize('estimator, build_dataset', metric_learners,
                         ids=ids_metric_learners)
def test_get_metric_is_independent_from_metric_learner(estimator,
                                                       build_dataset):
  """Tests that the get_metric method returns a function that is independent
  from the original metric learner"""
  input_data, labels, _, X = build_dataset()
  model = clone(estimator)
  set_random_state(model)

  # we fit the metric learner on it and then we compute the metric on some
  # points
  model.fit(*remove_y(model, input_data, labels))
  metric = model.get_metric()
  score = metric(X[0], X[1])

  # then we refit the estimator on another dataset
  model.fit(*remove_y(model, np.sin(input_data), labels))

  # we recompute the distance between the two points: it should be the same
  score_bis = metric(X[0], X[1])
  assert score_bis == score


@pytest.mark.parametrize('estimator, build_dataset', metric_learners,
                         ids=ids_metric_learners)
def test_get_metric_raises_error(estimator, build_dataset):
  """Tests that the metric returned by get_metric raises errors similar to
  the distance functions in scipy.spatial.distance"""
  input_data, labels, _, X = build_dataset()
  model = clone(estimator)
  set_random_state(model)
  model.fit(*remove_y(model, input_data, labels))
  metric = model.get_metric()

  list_test_get_metric_raises = [(X[0].tolist() + [5.2], X[1]),  # vectors with
                                 # different dimensions
                                 (X[0:4], X[1:5]),  # 2D vectors
                                 (X[0].tolist() + [5.2], X[1] + [7.2])]
  # vectors of same dimension but incompatible with what the metric learner
  # was trained on

  for u, v in list_test_get_metric_raises:
    with pytest.raises(ValueError):
      metric(u, v)


@pytest.mark.parametrize('estimator, build_dataset', metric_learners,
                         ids=ids_metric_learners)
def test_get_metric_works_does_not_raise(estimator, build_dataset):
  """Tests that the metric returned by get_metric does not raise errors (or
  warnings) similarly to the distance functions in scipy.spatial.distance"""
  input_data, labels, _, X = build_dataset()
  model = clone(estimator)
  set_random_state(model)
  model.fit(*remove_y(model, input_data, labels))
  metric = model.get_metric()

  list_test_get_metric_doesnt_raise = [(X[0], X[1]),
                                       (X[0].tolist(), X[1].tolist()),
                                       (X[0][None], X[1][None])]

  for u, v in list_test_get_metric_doesnt_raise:
    with warnings.catch_warnings(record=True) as record:
      metric(u, v)
    assert len(record) == 0

  # Test that the scalar case works
  model.components_ = np.array([3.1])
  metric = model.get_metric()
  for u, v in [(5, 6.7), ([5], [6.7]), ([[5]], [[6.7]])]:
    with warnings.catch_warnings(record=True) as record:
      metric(u, v)
    assert len(record) == 0


@pytest.mark.parametrize('estimator, build_dataset', metric_learners,
                         ids=ids_metric_learners)
def test_n_components(estimator, build_dataset):
  """Check that estimators that have a n_components parameters can use it
  and that it actually works as expected"""
  input_data, labels, _, X = build_dataset()
  model = clone(estimator)

  if hasattr(model, 'n_components'):
    set_random_state(model)
    model.set_params(n_components=None)
    model.fit(*remove_y(model, input_data, labels))
    assert model.components_.shape == (X.shape[1], X.shape[1])

    model = clone(estimator)
    set_random_state(model)
    model.set_params(n_components=X.shape[1] - 1)
    model.fit(*remove_y(model, input_data, labels))
    assert model.components_.shape == (X.shape[1] - 1, X.shape[1])

    model = clone(estimator)
    set_random_state(model)
    model.set_params(n_components=X.shape[1] + 1)
    with pytest.raises(ValueError) as expected_err:
      model.fit(*remove_y(model, input_data, labels))
    assert (str(expected_err.value) ==
            'Invalid n_components, must be in [1, {}]'.format(X.shape[1]))

    model = clone(estimator)
    set_random_state(model)
    model.set_params(n_components=0)
    with pytest.raises(ValueError) as expected_err:
      model.fit(*remove_y(model, input_data, labels))
    assert (str(expected_err.value) ==
            'Invalid n_components, must be in [1, {}]'.format(X.shape[1]))


@pytest.mark.parametrize('estimator, build_dataset', metric_learners,
                         ids=ids_metric_learners)
def test_score_pairs_warning(estimator, build_dataset):
  """Tests that score_pairs returns a FutureWarning regarding deprecation.
  Also that score_pairs and pair_distance have the same behaviour"""
  input_data, labels, _, X = build_dataset()
  model = clone(estimator)
  set_random_state(model)

  # We fit the metric learner on it and then we call score_pairs on some
  # points
  model.fit(*remove_y(model, input_data, labels))

  msg = ("score_pairs will be deprecated in release 0.7.0. "
         "Use pair_score to compute similarity scores, or "
         "pair_distances to compute distances.")
  with pytest.warns(FutureWarning) as raised_warning:
    score = model.score_pairs([[X[0], X[1]], ])
    dist = model.pair_distance([[X[0], X[1]], ])
    assert array_equal(score, dist)
  assert any([str(warning.message) == msg for warning in raised_warning])


if __name__ == '__main__':
  unittest.main()
