"""
Tests all functionality for Bilinear learners. Correctness, use cases,
warnings, etc.
"""
from itertools import product
import numpy as np
from numpy.testing import assert_array_almost_equal
import pytest
from sklearn import clone
from sklearn.datasets import make_spd_matrix
from sklearn.utils import check_random_state
from metric_learn.sklearn_shims import set_random_state
from test.test_utils import metric_learners_b, ids_metric_learners_b, \
  remove_y, IdentityBilinearLearner, build_classification

RNG = check_random_state(0)


@pytest.mark.parametrize('estimator, build_dataset', metric_learners_b,
                         ids=ids_metric_learners_b)
def test_same_similarity_with_two_methods(estimator, build_dataset):
  """"
  Tests that pair_score() and get_metric() give consistent results.
  In both cases, the results must match for the same input.
  Tests it for 'n_pairs' sampled from 'n' d-dimentional arrays.
  """
  input_data, labels, _, X = build_dataset()
  n_samples = 20
  X = X[:n_samples]
  model = clone(estimator)
  set_random_state(model)
  model.fit(*remove_y(estimator, input_data, labels))
  random_pairs = np.array(list(product(X, X)))

  dist1 = model.pair_score(random_pairs)
  dist2 = [model.get_metric()(p[0], p[1]) for p in random_pairs]

  assert_array_almost_equal(dist1, dist2)


@pytest.mark.parametrize('estimator, build_dataset', metric_learners_b,
                         ids=ids_metric_learners_b)
def test_check_correctness_similarity(estimator, build_dataset):
  """
  Tests the correctness of the results made from socre_paris(),
  get_metric() and get_bilinear_matrix. Results are compared with
  the real bilinear similarity calculated in-place.
  """
  input_data, labels, _, X = build_dataset()
  n_samples = 20
  X = X[:n_samples]
  model = clone(estimator)
  set_random_state(model)
  model.fit(*remove_y(estimator, input_data, labels))
  random_pairs = np.array(list(product(X, X)))

  dist1 = model.pair_score(random_pairs)
  dist2 = [model.get_metric()(p[0], p[1]) for p in random_pairs]
  dist3 = [np.dot(np.dot(p[0].T, model.get_bilinear_matrix()), p[1])
           for p in random_pairs]
  desired = [np.dot(np.dot(p[0].T, model.components_), p[1])
             for p in random_pairs]

  assert_array_almost_equal(dist1, desired)  # pair_score
  assert_array_almost_equal(dist2, desired)  # get_metric
  assert_array_almost_equal(dist3, desired)  # get_metric


# This is a `hardcoded` handmade tests, to make sure the computation
# made at BilinearMixin is correct.
def test_check_handmade_example():
  """
  Checks that pair_score() result is correct comparing it with a
  handmade example.
  """
  u = np.array([0, 1, 2])
  v = np.array([3, 4, 5])
  mixin = IdentityBilinearLearner()
  mixin.fit([u, v], [0, 0])  # Identity fit
  c = np.array([[2, 4, 6], [6, 4, 2], [1, 2, 3]])
  mixin.components_ = c  # Force components_
  dists = mixin.pair_score([[u, v], [v, u]])
  assert_array_almost_equal(dists, [96, 120])


# Note: This test needs to be `hardcoded` as the similarity martix must
# be symmetric. Running on all Bilinear learners will throw an error as
# the matrix can be non-symmetric.
def test_check_handmade_symmetric_example():
  """
  When the bilinear matrix is the identity. The similarity
  between two arrays must be equal: S(u,v) = S(v,u). Also
  checks the random case: when the matrix is spd and symetric.
  """
  input_data, labels, _, X = build_classification()
  n_samples = 20
  X = X[:n_samples]
  model = clone(IdentityBilinearLearner())  # Identity matrix
  set_random_state(model)
  model.fit(*remove_y(IdentityBilinearLearner(), input_data, labels))
  random_pairs = np.array(list(product(X, X)))

  pairs_reverse = [[p[1], p[0]] for p in random_pairs]
  dist1 = model.pair_score(random_pairs)
  dist2 = model.pair_score(pairs_reverse)
  assert_array_almost_equal(dist1, dist2)

  # Random pairs for M = spd Matrix
  spd_matrix = make_spd_matrix(X[0].shape[-1], random_state=RNG)
  model.components_ = spd_matrix
  dist1 = model.pair_score(random_pairs)
  dist2 = model.pair_score(pairs_reverse)
  assert_array_almost_equal(dist1, dist2)


@pytest.mark.parametrize('estimator, build_dataset', metric_learners_b,
                         ids=ids_metric_learners_b)
def test_pair_score_finite(estimator, build_dataset):
  """
  Checks for 'n' pair_score() of 'd' dimentions, that all
  similarities are finite numbers: not NaN, +inf or -inf.
  Considers a random M for bilinear similarity.
  """
  input_data, labels, _, X = build_dataset()
  n_samples = 20
  X = X[:n_samples]
  model = clone(estimator)
  set_random_state(model)
  model.fit(*remove_y(estimator, input_data, labels))
  random_pairs = np.array(list(product(X, X)))
  dist1 = model.pair_score(random_pairs)
  assert np.isfinite(dist1).all()


@pytest.mark.parametrize('estimator, build_dataset', metric_learners_b,
                         ids=ids_metric_learners_b)
def test_check_error_with_pair_distance(estimator, build_dataset):
  """
  Check that calling `pair_distance` is not possible with a Bilinear learner.
  An Exception must be shown instead.
  """
  input_data, labels, _, X = build_dataset()
  model = clone(estimator)
  set_random_state(model)
  model.fit(*remove_y(model, input_data, labels))
  random_pairs = np.array(list(product(X, X)))

  msg = ("This learner does not learn a distance, thus ",
         "this method is not implemented. Use pair_score instead")
  with pytest.raises(Exception) as e:
    _ = model.pair_distance(random_pairs)
  assert e.value.args[0] == msg