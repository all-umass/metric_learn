import pytest
import unittest
from sklearn.utils.estimator_checks import check_estimator
from sklearn.base import TransformerMixin
from sklearn.pipeline import make_pipeline
from sklearn.utils import check_random_state
from sklearn.utils.estimator_checks import is_public_parameter
from sklearn.utils.testing import (assert_allclose_dense_sparse,
                                   set_random_state)

from metric_learn import (Covariance, LFDA, LMNN, MLKR, NCA,
                          ITML_Supervised, LSML_Supervised,
                          MMC_Supervised, RCA_Supervised, SDML_Supervised)
from sklearn import clone
import numpy as np
from sklearn.model_selection import (cross_val_score, cross_val_predict,
                                     train_test_split, KFold)
from sklearn.utils.testing import _get_args
from test.test_utils import (metric_learners, ids_metric_learners,
                             mock_preprocessor)


# Wrap the _Supervised methods with a deterministic wrapper for testing.
class deterministic_mixin(object):
  def fit(self, X, y):
    rs = np.random.RandomState(1234)
    return super(deterministic_mixin, self).fit(X, y, random_state=rs)


class dLSML(deterministic_mixin, LSML_Supervised):
  pass


class dITML(deterministic_mixin, ITML_Supervised):
  pass


class dMMC(deterministic_mixin, MMC_Supervised):
  pass


class dSDML(deterministic_mixin, SDML_Supervised):
  pass


class dRCA(deterministic_mixin, RCA_Supervised):
  pass


class TestSklearnCompat(unittest.TestCase):
  def test_covariance(self):
    check_estimator(Covariance)

  def test_lmnn(self):
    check_estimator(LMNN)

  def test_lfda(self):
    check_estimator(LFDA)

  def test_mlkr(self):
    check_estimator(MLKR)

  def test_nca(self):
    check_estimator(NCA)

  def test_lsml(self):
    check_estimator(dLSML)

  def test_itml(self):
    check_estimator(dITML)

  def test_mmc(self):
    check_estimator(dMMC)

  # This fails due to a FloatingPointError
  # def test_sdml(self):
  #   check_estimator(dSDML)

  # This fails because the default num_chunks isn't data-dependent.
  # def test_rca(self):
  #   check_estimator(RCA_Supervised)


RNG = check_random_state(0)


# ---------------------- Test scikit-learn compatibility ----------------------


@pytest.mark.parametrize('with_preprocessor', [True, False])
@pytest.mark.parametrize('estimator, build_dataset', metric_learners,
                         ids=ids_metric_learners)
def test_cross_validation_is_finite(estimator, build_dataset,
                                    with_preprocessor):
  """Tests that validation on metric-learn estimators returns something finite
  """
  if any(hasattr(estimator, method) for method in ["predict", "score"]):
    input_data, labels, preprocessor, _ = build_dataset(with_preprocessor)
    estimator = clone(estimator)
    estimator.set_params(preprocessor=preprocessor)
    set_random_state(estimator)
    if hasattr(estimator, "score"):
      assert np.isfinite(cross_val_score(estimator, input_data, labels)).all()
    if hasattr(estimator, "predict"):
      assert np.isfinite(cross_val_predict(estimator,
                                           input_data, labels)).all()


@pytest.mark.parametrize('with_preprocessor', [True, False])
@pytest.mark.parametrize('estimator, build_dataset', metric_learners,
                         ids=ids_metric_learners)
def test_cross_validation_manual_vs_scikit(estimator, build_dataset,
                                           with_preprocessor):
  """Tests that if we make a manual cross-validation, the result will be the
  same as scikit-learn's cross-validation (some code for generating the
  folds is taken from scikit-learn).
  """
  if any(hasattr(estimator, method) for method in ["predict", "score"]):
    input_data, labels, preprocessor, _ = build_dataset(with_preprocessor)
    estimator = clone(estimator)
    estimator.set_params(preprocessor=preprocessor)
    set_random_state(estimator)
    n_splits = 3
    kfold = KFold(shuffle=False, n_splits=n_splits)
    n_samples = input_data.shape[0]
    fold_sizes = (n_samples // n_splits) * np.ones(n_splits, dtype=np.int)
    fold_sizes[:n_samples % n_splits] += 1
    current = 0
    scores, predictions = [], np.zeros(input_data.shape[0])
    for fold_size in fold_sizes:
      start, stop = current, current + fold_size
      current = stop
      test_slice = slice(start, stop)
      train_mask = np.ones(input_data.shape[0], bool)
      train_mask[test_slice] = False
      y_train, y_test = labels[train_mask], labels[test_slice]
      estimator.fit(input_data[train_mask], y_train)
      if hasattr(estimator, "score"):
        scores.append(estimator.score(input_data[test_slice], y_test))
      if hasattr(estimator, "predict"):
        predictions[test_slice] = estimator.predict(input_data[test_slice])
    if hasattr(estimator, "score"):
      assert all(scores == cross_val_score(estimator, input_data, labels,
                                           cv=kfold))
    if hasattr(estimator, "predict"):
      assert all(predictions == cross_val_predict(estimator, input_data,
                                                  labels,
                                                  cv=kfold))


def check_score(estimator, tuples, y):
  if hasattr(estimator, "score"):
    score = estimator.score(tuples, y)
    assert np.isfinite(score)


def check_predict(estimator, tuples):
  if hasattr(estimator, "predict"):
    y_predicted = estimator.predict(tuples)
    assert len(y_predicted), len(tuples)


@pytest.mark.parametrize('with_preprocessor', [True, False])
@pytest.mark.parametrize('estimator, build_dataset', metric_learners,
                         ids=ids_metric_learners)
def test_simple_estimator(estimator, build_dataset, with_preprocessor):
  """Tests that fit, predict and scoring works.
  """
  if any(hasattr(estimator, method) for method in ["predict", "score"]):
    input_data, labels, preprocessor, _ = build_dataset(with_preprocessor)
    (tuples_train, tuples_test, y_train,
     y_test) = train_test_split(input_data, labels, random_state=RNG)
    estimator = clone(estimator)
    estimator.set_params(preprocessor=preprocessor)
    set_random_state(estimator)

    estimator.fit(tuples_train, y_train)
    check_score(estimator, tuples_test, y_test)
    check_predict(estimator, tuples_test)


@pytest.mark.parametrize('estimator', [est[0] for est in metric_learners],
                         ids=ids_metric_learners)
@pytest.mark.parametrize('preprocessor', [None, mock_preprocessor])
def test_no_attributes_set_in_init(estimator, preprocessor):
  """Check setting during init. Adapted from scikit-learn."""
  estimator = clone(estimator)
  estimator.set_params(preprocessor=preprocessor)
  if hasattr(type(estimator).__init__, "deprecated_original"):
      return

  init_params = _get_args(type(estimator).__init__)
  parents_init_params = [param for params_parent in
                         (_get_args(parent) for parent in
                          type(estimator).__mro__)
                         for param in params_parent]

  # Test for no setting apart from parameters during init
  invalid_attr = (set(vars(estimator)) - set(init_params) -
                  set(parents_init_params))
  assert not invalid_attr, \
      ("Estimator %s should not set any attribute apart"
       " from parameters during init. Found attributes %s."
       % (type(estimator).__name__, sorted(invalid_attr)))
  # Ensure that each parameter is set in init
  invalid_attr = (set(init_params) - set(vars(estimator)) -
                  set(["self"]))
  assert not invalid_attr, \
      ("Estimator %s should store all parameters"
       " as an attribute during init. Did not find "
       "attributes %s." % (type(estimator).__name__, sorted(invalid_attr)))


@pytest.mark.parametrize('with_preprocessor', [True, False])
@pytest.mark.parametrize('estimator, build_dataset', metric_learners,
                         ids=ids_metric_learners)
def test_estimators_fit_returns_self(estimator, build_dataset,
                                     with_preprocessor):
  """Check if self is returned when calling fit"""
  # Adapted from scikit-learn
  input_data, labels, preprocessor, _ = build_dataset(with_preprocessor)
  estimator = clone(estimator)
  estimator.set_params(preprocessor=preprocessor)
  assert estimator.fit(input_data, labels) is estimator


@pytest.mark.parametrize('with_preprocessor', [True, False])
@pytest.mark.parametrize('estimator, build_dataset', metric_learners,
                         ids=ids_metric_learners)
def test_pipeline_consistency(estimator, build_dataset,
                              with_preprocessor):
  # Adapted from scikit learn
  # check that make_pipeline(est) gives same score as est
  input_data, y, preprocessor, _ = build_dataset(with_preprocessor)

  def make_random_state(estimator, in_pipeline):
    rs = {}
    name_estimator = estimator.__class__.__name__
    if name_estimator[-11:] == '_Supervised':
      name_param = 'random_state'
      if in_pipeline:
          name_param = name_estimator.lower() + '__' + name_param
      rs[name_param] = check_random_state(0)
    return rs

  estimator = clone(estimator)
  estimator.set_params(preprocessor=preprocessor)
  pipeline = make_pipeline(estimator)
  estimator.fit(input_data, y, **make_random_state(estimator, False))
  pipeline.fit(input_data, y, **make_random_state(estimator, True))

  if hasattr(estimator, 'score'):
    result = estimator.score(input_data, y)
    result_pipe = pipeline.score(input_data, y)
    assert_allclose_dense_sparse(result, result_pipe)

  if hasattr(estimator, 'predict'):
    result = estimator.predict(input_data)
    result_pipe = pipeline.predict(input_data)
    assert_allclose_dense_sparse(result, result_pipe)

  if issubclass(estimator.__class__, TransformerMixin):
    if hasattr(estimator, 'transform'):
      result = estimator.transform(input_data)
      result_pipe = pipeline.transform(input_data)
      assert_allclose_dense_sparse(result, result_pipe)


@pytest.mark.parametrize('with_preprocessor',[True, False])
@pytest.mark.parametrize('estimator, build_dataset', metric_learners,
                         ids=ids_metric_learners)
def test_dict_unchanged(estimator, build_dataset, with_preprocessor):
  # Adapted from scikit-learn
  (input_data, labels, preprocessor,
   to_transform) = build_dataset(with_preprocessor)
  estimator = clone(estimator)
  estimator.set_params(preprocessor=preprocessor)
  if hasattr(estimator, "num_dims"):
    estimator.num_dims = 1
  estimator.fit(input_data, labels)

  def check_dict():
    assert estimator.__dict__ == dict_before, (
        "Estimator changes __dict__ during %s" % method)
  for method in ["predict", "decision_function", "predict_proba"]:
    if hasattr(estimator, method):
      dict_before = estimator.__dict__.copy()
      getattr(estimator, method)(input_data)
      check_dict()
  if hasattr(estimator, "transform"):
    dict_before = estimator.__dict__.copy()
    # we transform only dataset of points
    estimator.transform(to_transform)
    check_dict()


@pytest.mark.parametrize('with_preprocessor',[True, False])
@pytest.mark.parametrize('estimator, build_dataset', metric_learners,
                         ids=ids_metric_learners)
def test_dont_overwrite_parameters(estimator, build_dataset,
                                   with_preprocessor):
  # Adapted from scikit-learn
  # check that fit method only changes or sets private attributes
  input_data, labels, preprocessor, _ = build_dataset(with_preprocessor)
  estimator = clone(estimator)
  estimator.set_params(preprocessor=preprocessor)
  if hasattr(estimator, "num_dims"):
    estimator.num_dims = 1
  dict_before_fit = estimator.__dict__.copy()

  estimator.fit(input_data, labels)
  dict_after_fit = estimator.__dict__

  public_keys_after_fit = [key for key in dict_after_fit.keys()
                           if is_public_parameter(key)]

  attrs_added_by_fit = [key for key in public_keys_after_fit
                        if key not in dict_before_fit.keys()]

  # check that fit doesn't add any public attribute
  assert not attrs_added_by_fit, (
      "Estimator adds public attribute(s) during"
      " the fit method."
      " Estimators are only allowed to add private "
      "attributes"
      " either started with _ or ended"
      " with _ but %s added" % ', '.join(attrs_added_by_fit))

  # check that fit doesn't change any public attribute
  attrs_changed_by_fit = [key for key in public_keys_after_fit
                          if (dict_before_fit[key]
                              is not dict_after_fit[key])]

  assert not attrs_changed_by_fit, (
      "Estimator changes public attribute(s) during"
      " the fit method. Estimators are only allowed"
      " to change attributes started"
      " or ended with _, but"
      " %s changed" % ', '.join(attrs_changed_by_fit))


if __name__ == '__main__':
  unittest.main()
