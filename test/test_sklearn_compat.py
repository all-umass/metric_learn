import pytest
import unittest
from sklearn.utils.estimator_checks import check_estimator
from sklearn.base import TransformerMixin
from sklearn.pipeline import make_pipeline
from sklearn.utils import check_random_state
from metric_learn.sklearn_shims import (assert_allclose_dense_sparse,
                                        set_random_state, _get_args,
                                        is_public_parameter, get_scorer)
from metric_learn import (Covariance, LFDA, LMNN_Supervised, MLKR, NCA,
                          ITML_Supervised, LSML_Supervised,
                          MMC_Supervised, RCA_Supervised, SDML_Supervised,
                          SCML_Supervised)
from sklearn import clone
import numpy as np
from sklearn.model_selection import (cross_val_score, cross_val_predict,
                                     train_test_split, KFold)
from test.test_utils import (metric_learners, ids_metric_learners,
                             mock_preprocessor, tuples_learners,
                             ids_tuples_learners, pairs_learners,
                             ids_pairs_learners, remove_y,
                             metric_learners_pipeline,
                             ids_metric_learners_pipeline)


class Stable_RCA_Supervised(RCA_Supervised):

  def __init__(self, n_components=None,
               chunk_size=2, preprocessor=None, random_state=None):
    # this init makes RCA stable for scikit-learn examples.
    super(Stable_RCA_Supervised, self).__init__(
        n_chunks=2, n_components=n_components,
        chunk_size=chunk_size, preprocessor=preprocessor,
        random_state=random_state)


class Stable_SDML_Supervised(SDML_Supervised):

  def __init__(self, sparsity_param=0.01,
               n_constraints=None, verbose=False, preprocessor=None,
               random_state=None):
    # this init makes SDML stable for scikit-learn examples.
    super(Stable_SDML_Supervised, self).__init__(
        sparsity_param=sparsity_param,
        n_constraints=n_constraints, verbose=verbose,
        preprocessor=preprocessor, balance_param=1e-5, prior='identity',
        random_state=random_state)


class TestSklearnCompat(unittest.TestCase):
  def test_covariance(self):
    check_estimator(Covariance())

  def test_lmnn(self):
    check_estimator(LMNN_Supervised())

  def test_lfda(self):
    check_estimator(LFDA())

  def test_mlkr(self):
    check_estimator(MLKR())

  def test_nca(self):
    check_estimator(NCA())

  def test_lsml(self):
    check_estimator(LSML_Supervised())

  def test_itml(self):
    check_estimator(ITML_Supervised())

  def test_mmc(self):
    check_estimator(MMC_Supervised())

  def test_sdml(self):
    check_estimator(Stable_SDML_Supervised())

  def test_rca(self):
    check_estimator(Stable_RCA_Supervised())

  def test_scml(self):
    msg = "As no value for `n_basis` was selected, "
    with pytest.warns(UserWarning) as raised_warning:
      check_estimator(SCML_Supervised())
    assert msg in str(raised_warning[0].message)


RNG = check_random_state(0)


# ---------------------- Test scikit-learn compatibility ----------------------

def generate_array_like(input_data, labels=None):
  """Helper function to generate array-like variants of numpy datasets,
  for testing purposes."""
  list_data = input_data.tolist()
  input_data_changed = [input_data, list_data, tuple(list_data)]
  if input_data.ndim >= 2:
    input_data_changed.append(tuple(tuple(x) for x in list_data))
  if input_data.ndim >= 3:
    input_data_changed.append(tuple(tuple(tuple(x) for x in y) for y in
                                    list_data))
  if input_data.ndim == 2:
    pd = pytest.importorskip('pandas')
    input_data_changed.append(pd.DataFrame(input_data))
  if labels is not None:
    labels_changed = [labels, list(labels), tuple(labels)]
  else:
    labels_changed = [labels]
  return input_data_changed, labels_changed


@pytest.mark.integration
@pytest.mark.parametrize('with_preprocessor', [True, False])
@pytest.mark.parametrize('estimator, build_dataset', metric_learners,
                         ids=ids_metric_learners)
def test_array_like_inputs(estimator, build_dataset, with_preprocessor):
  """Test that metric-learners can have as input (of all functions that are
  applied on data) any array-like object."""
  input_data, labels, preprocessor, X = build_dataset(with_preprocessor)

  # we subsample the data for the test to be more efficient
  input_data, _, labels, _ = train_test_split(input_data, labels,
                                              train_size=40,
                                              random_state=42)
  X = X[:10]

  estimator = clone(estimator)
  estimator.set_params(preprocessor=preprocessor)
  set_random_state(estimator)
  input_variants, label_variants = generate_array_like(input_data, labels)
  for input_variant in input_variants:
    for label_variant in label_variants:
      estimator.fit(*remove_y(estimator, input_variant, label_variant))
    if hasattr(estimator, "predict"):
      estimator.predict(input_variant)
    if hasattr(estimator, "predict_proba"):
      estimator.predict_proba(input_variant)  # anticipation in case some
      # time we have that, or if ppl want to contribute with new algorithms
      # it will be checked automatically
    if hasattr(estimator, "decision_function"):
      estimator.decision_function(input_variant)
    if hasattr(estimator, "score"):
      for label_variant in label_variants:
        estimator.score(*remove_y(estimator, input_variant, label_variant))

  X_variants, _ = generate_array_like(X)
  for X_variant in X_variants:
    estimator.transform(X_variant)

  pairs = np.array([[X[0], X[1]], [X[0], X[2]]])
  pairs_variants, _ = generate_array_like(pairs)

  not_implemented_msg = ""
  # Todo in 0.7.0: Change 'not_implemented_msg' for the message that says
  # "This learner does not have pair_distance"

  for pairs_variant in pairs_variants:
    estimator.pair_score(pairs_variant)  # All learners have pair_score

    # But not all of them will have pair_distance
    try:
      estimator.pair_distance(pairs_variant)
    except Exception as raised_exception:
      assert raised_exception.value.args[0] == not_implemented_msg


@pytest.mark.parametrize('with_preprocessor', [True, False])
@pytest.mark.parametrize('estimator, build_dataset', pairs_learners,
                         ids=ids_pairs_learners)
def test_various_scoring_on_tuples_learners(estimator, build_dataset,
                                            with_preprocessor):
  """Tests that scikit-learn's scoring returns something finite,
  for other scoring than default scoring. (List of scikit-learn's scores can be
  found in sklearn.metrics._scorer). For each type of output (predict,
  predict_proba, decision_function), we test a bunch of scores.
  We only test on pairs learners because quadruplets don't have a y argument.
  """
  input_data, labels, preprocessor, _ = build_dataset(with_preprocessor)
  estimator = clone(estimator)
  estimator.set_params(preprocessor=preprocessor)
  set_random_state(estimator)

  # scores that need a predict function: every tuples learner should have a
  # predict function (whether the pair is of positive samples or negative
  # samples)
  for scoring in ['accuracy', 'f1']:
    check_score_is_finite(scoring, estimator, input_data, labels)
  # scores that need a predict_proba:
  if hasattr(estimator, "predict_proba"):
    for scoring in ['neg_log_loss', 'brier_score']:
      check_score_is_finite(scoring, estimator, input_data, labels)
  # scores that need a decision_function: every tuples learner should have a
  # decision function (the metric between points)
  for scoring in ['roc_auc', 'average_precision', 'precision', 'recall']:
    check_score_is_finite(scoring, estimator, input_data, labels)


def check_score_is_finite(scoring, estimator, input_data, labels):
  estimator = clone(estimator)
  assert np.isfinite(cross_val_score(estimator, input_data, labels,
                                     scoring=scoring)).all()
  estimator.fit(input_data, labels)
  assert np.isfinite(get_scorer(scoring)(estimator, input_data, labels))


@pytest.mark.parametrize('estimator, build_dataset', tuples_learners,
                         ids=ids_tuples_learners)
def test_cross_validation_is_finite(estimator, build_dataset):
  """Tests that validation on metric-learn estimators returns something finite
  """
  input_data, labels, preprocessor, _ = build_dataset()
  estimator = clone(estimator)
  estimator.set_params(preprocessor=preprocessor)
  set_random_state(estimator)
  assert np.isfinite(cross_val_score(estimator,
                                     *remove_y(estimator, input_data, labels)
                                     )).all()
  assert np.isfinite(cross_val_predict(estimator,
                                       *remove_y(estimator, input_data, labels)
                                       )).all()


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
    fold_sizes = (n_samples // n_splits) * np.ones(n_splits, dtype=np.int64)
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
      estimator.fit(*remove_y(estimator, input_data[train_mask], y_train))
      if hasattr(estimator, "score"):
        scores.append(estimator.score(*remove_y(
            estimator, input_data[test_slice], y_test)))
      if hasattr(estimator, "predict"):
        predictions[test_slice] = estimator.predict(input_data[test_slice])
    if hasattr(estimator, "score"):
      assert all(scores == cross_val_score(
          estimator, *remove_y(estimator, input_data, labels),
          cv=kfold))
    if hasattr(estimator, "predict"):
      assert all(predictions == cross_val_predict(
          estimator,
          *remove_y(estimator, input_data, labels),
          cv=kfold))


def check_score(estimator, tuples, y):
  if hasattr(estimator, "score"):
    score = estimator.score(*remove_y(estimator, tuples, y))
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

    estimator.fit(*remove_y(estimator, tuples_train, y_train))
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
  assert estimator.fit(*remove_y(estimator, input_data, labels)) is estimator


@pytest.mark.parametrize('with_preprocessor', [True, False])
@pytest.mark.parametrize('estimator, build_dataset', metric_learners_pipeline,
                         ids=ids_metric_learners_pipeline)
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
  estimator.set_params(preprocessor=preprocessor,
                       **make_random_state(estimator, False))
  pipeline = make_pipeline(estimator)
  estimator.fit(input_data, y)
  estimator.set_params(preprocessor=preprocessor)
  pipeline.set_params(**make_random_state(estimator, True))
  pipeline.fit(input_data, y)

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


@pytest.mark.parametrize('with_preprocessor', [True, False])
@pytest.mark.parametrize('estimator, build_dataset', metric_learners,
                         ids=ids_metric_learners)
def test_dict_unchanged(estimator, build_dataset, with_preprocessor):
  # Adapted from scikit-learn
  (input_data, labels, preprocessor,
   to_transform) = build_dataset(with_preprocessor)
  estimator = clone(estimator)
  estimator.set_params(preprocessor=preprocessor)
  if hasattr(estimator, "n_components"):
    estimator.n_components = 1
  estimator.fit(*remove_y(estimator, input_data, labels))

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


@pytest.mark.parametrize('with_preprocessor', [True, False])
@pytest.mark.parametrize('estimator, build_dataset', metric_learners,
                         ids=ids_metric_learners)
def test_dont_overwrite_parameters(estimator, build_dataset,
                                   with_preprocessor):
  # Adapted from scikit-learn
  # check that fit method only changes or sets private attributes
  input_data, labels, preprocessor, _ = build_dataset(with_preprocessor)
  estimator = clone(estimator)
  estimator.set_params(preprocessor=preprocessor)
  if hasattr(estimator, "n_components"):
    estimator.n_components = 1
  dict_before_fit = estimator.__dict__.copy()

  estimator.fit(*remove_y(estimator, input_data, labels))
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
