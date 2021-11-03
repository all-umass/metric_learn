"""
Tests preprocesor, warnings, errors. Also made util functions to build datasets
in a general way for each learner. Here is also the list of learners of each
kind that are used as a parameters in tests in other files. Util functions.
"""
import pytest
from scipy.linalg import eigh, pinvh
from collections import namedtuple
import numpy as np
from numpy.testing import assert_array_equal, assert_equal, assert_raises
from sklearn.model_selection import train_test_split
from sklearn.utils import check_random_state, shuffle
from metric_learn.sklearn_shims import set_random_state
from sklearn.base import clone
from metric_learn._util import (check_input, make_context, preprocess_tuples,
                                make_name, preprocess_points,
                                check_collapsed_pairs, validate_vector,
                                _check_sdp_from_eigen, _check_n_components,
                                check_y_valid_values_for_pairs,
                                _auto_select_init, _pseudo_inverse_from_eig,
                                _get_random_indices,
                                _initialize_similarity_bilinear)
from metric_learn import (ITML, LSML, MMC, RCA, SDML, Covariance, LFDA,
                          LMNN, MLKR, NCA, ITML_Supervised, LSML_Supervised,
                          MMC_Supervised, RCA_Supervised, SDML_Supervised,
                          SCML, SCML_Supervised, Constraints)
from metric_learn.base_metric import (ArrayIndexer, MahalanobisMixin,
                                      BilinearMixin,
                                      _PairsClassifierMixin,
                                      _TripletsClassifierMixin,
                                      _QuadrupletsClassifierMixin)
from metric_learn.exceptions import PreprocessorError, NonPSDError
from sklearn.datasets import make_regression, make_blobs, load_iris


SEED = 42
RNG = check_random_state(SEED)


# -------------------- Mock classes for testing ------------------------


class RandomBilinearLearner(BilinearMixin):
  """A simple Random bilinear mixin that returns an random matrix
  M as learned. Class for testing purposes.
  """
  def __init__(self, init='random', preprocessor=None, random_state=33):
    super().__init__(preprocessor=preprocessor)
    self.init = init
    self.random_state = random_state

  def fit(self, X, y):
    """
    Checks input's format. A random (d,d) matrix is set.
    """
    X, y = self._prepare_inputs(X, y, ensure_min_samples=2)
    self.d_ = np.shape(X[0])[-1]
    M = _initialize_similarity_bilinear(X,
                                        init=self.init,
                                        strict_pd=False,
                                        random_state=self.random_state)
    self.components_ = M
    return self


class IdentityBilinearLearner(BilinearMixin):
  """A simple Identity bilinear mixin that returns an identity matrix
  M as learned. Class for testing purposes.
  """
  def __init__(self, init='identity', preprocessor=None, random_state=33):
    super().__init__(preprocessor=preprocessor)
    self.init = init
    self.random_state = random_state

  def fit(self, X, y):
    """
    Checks input's format. Sets M matrix to identity of shape (d,d)
    where d is the dimension of the input.
    """
    X, y = self._prepare_inputs(X, y, ensure_min_samples=2)
    self.d_ = np.shape(X[0])[-1]
    M = _initialize_similarity_bilinear(X,
                                        init=self.init,
                                        strict_pd=False,
                                        random_state=self.random_state)
    self.components_ = M
    return self


class MockPairIdentityBilinearLearner(BilinearMixin,
                                      _PairsClassifierMixin):

  def __init__(self, init='identity', preprocessor=None, random_state=33):
      super().__init__(preprocessor=preprocessor)
      self.init = init
      self.random_state = random_state

  def fit(self, pairs, y, calibration_params=None):
    calibration_params = (calibration_params if calibration_params is not
                          None else dict())
    self._validate_calibration_params(**calibration_params)
    pairs = self._prepare_inputs(pairs, type_of_inputs='tuples')
    self.d_ = np.shape(pairs[0][0])[-1]
    M = _initialize_similarity_bilinear(pairs,
                                        init=self.init,
                                        strict_pd=False,
                                        random_state=self.random_state)
    self.components_ = M
    self.calibrate_threshold(pairs, y, **calibration_params)
    return self


class MockTripletsIdentityBilinearLearner(BilinearMixin,
                                          _TripletsClassifierMixin):

  def __init__(self, init='identity', preprocessor=None, random_state=33):
      super().__init__(preprocessor=preprocessor)
      self.init = init
      self.random_state = random_state

  def fit(self, triplets):
    triplets = self._prepare_inputs(triplets, type_of_inputs='tuples')
    self.d_ = np.shape(triplets[0][0])[-1]
    M = _initialize_similarity_bilinear(triplets,
                                        init=self.init,
                                        strict_pd=False,
                                        random_state=self.random_state)
    self.components_ = M
    return self


class MockQuadrpletsIdentityBilinearLearner(BilinearMixin,
                                            _QuadrupletsClassifierMixin):

  def __init__(self, init='identity', preprocessor=None, random_state=33):
      super().__init__(preprocessor=preprocessor)
      self.init = init
      self.random_state = random_state

  def fit(self, quadruplets):
    quadruplets = self._prepare_inputs(quadruplets, type_of_inputs='tuples')
    self.d_ = np.shape(quadruplets[0][0])[-1]
    M = _initialize_similarity_bilinear(quadruplets,
                                        init=self.init,
                                        strict_pd=False,
                                        random_state=self.random_state)
    self.components_ = M
    return self


# ------------------ Building dummy data for learners ------------------

Dataset = namedtuple('Dataset', ('data target preprocessor to_transform'))
# Data and target are what we will fit on. Preprocessor is the additional
# data if we use a preprocessor (which should be the default ArrayIndexer),
# and to_transform is some additional data that we would want to transform


def build_classification(with_preprocessor=False):
  """
  Basic array 'X, y' for testing when using a preprocessor, for instance,
  for clustering. For supervised learners.

  If no preprocesor: 'data' are raw points, 'target' are dummy labels,
  'preprocesor' is None, and 'to_transform' are points.

  If preprocessor: 'data' are point indices, 'target' are dummy labels,
  'preprocessor' are unique points, 'to_transform' are points.
  """
  X, y = shuffle(*make_blobs(random_state=SEED),
                 random_state=SEED)
  indices = shuffle(np.arange(X.shape[0]), random_state=SEED).astype(int)
  if with_preprocessor:
    return Dataset(indices, y[indices], X, indices)
  else:
    return Dataset(X[indices], y[indices], None, X[indices])


def build_regression(with_preprocessor=False):
  """
  Basic array 'X, y' for testing when using a preprocessor, for regression.
  For supervised learners.

  If no preprocesor: 'data' are raw points, 'target' are dummy labels,
  'preprocesor' is None, and 'to_transform' are points.

  If preprocessor: 'data' are point indices, 'target' are dummy labels,
  'preprocessor' are unique points, 'to_transform' are points.
  """
  X, y = shuffle(*make_regression(n_samples=100, n_features=5,
                                  random_state=SEED),
                 random_state=SEED)
  indices = shuffle(np.arange(X.shape[0]), random_state=SEED).astype(int)
  if with_preprocessor:
    return Dataset(indices, y[indices], X, indices)
  else:
    return Dataset(X[indices], y[indices], None, X[indices])


def build_data():
  """Aux function: Returns 'X, pairs' taken from the iris dataset, where
  pairs are positive and negative pairs for PairClassifiers."""
  input_data, labels = load_iris(return_X_y=True)
  X, y = shuffle(input_data, labels, random_state=SEED)
  num_constraints = 50
  constraints = Constraints(y)
  pairs = (
      constraints
      .positive_negative_pairs(num_constraints, same_length=True,
                               random_state=check_random_state(SEED)))
  return X, pairs


def build_pairs(with_preprocessor=False):
  """
  For all pair weakly-supervised learners.

  Returns: data, target, preprocessor, to_transform.

  If no preprocesor: 'data' are raw pairs, 'target' are dummy labels,
  'preprocesor' is None, and 'to_transform' are points.

  If preprocessor: 'data' are pair indices, 'target' are dummy labels,
  'preprocessor' are unique points, 'to_transform' are points.
  """
  X, indices = build_data()
  c = np.vstack([np.column_stack(indices[:2]), np.column_stack(indices[2:])])
  target = np.concatenate([np.ones(indices[0].shape[0]),
                           - np.ones(indices[0].shape[0])])
  c, target = shuffle(c, target, random_state=SEED)
  if with_preprocessor:
    # if preprocessor, we build a 2D array of pairs of indices
    return Dataset(c, target, X, c[:, 0])
  else:
    # if not, we build a 3D array of pairs of samples
    return Dataset(X[c], target, None, X[c[:, 0]])


def build_triplets(with_preprocessor=False):
  """
  For all triplet weakly-supervised learners.

  Returns: data, target, preprocessor, to_transform.

  If no preprocesor: 'data' are raw triplets, 'target' are dummy labels,
  'preprocesor' is None, and 'to_transform' are points.

  If preprocessor: 'data' are triplets indices, 'target' are dummy labels,
  'preprocessor' are unique points, 'to_transform' are points.
  """
  input_data, labels = load_iris(return_X_y=True)
  X, y = shuffle(input_data, labels, random_state=SEED)
  constraints = Constraints(y)
  triplets = constraints.generate_knntriplets(X, k_genuine=3, k_impostor=4)
  if with_preprocessor:
    # if preprocessor, we build a 2D array of triplets of indices
    return Dataset(triplets, np.ones(len(triplets)), X, np.arange(len(X)))
  else:
    # if not, we build a 3D array of triplets of samples
    return Dataset(X[triplets], np.ones(len(triplets)), None, X)


def build_quadruplets(with_preprocessor=False):
  """
  For all Quadruplets weakly-supervised learners.

  Returns: data, target, preprocessor, to_transform.

  If no preprocesor: 'data' are raw quadruplets, 'target' are dummy labels,
  'preprocesor' is None, and 'to_transform' are points.

  If preprocessor: 'data' are quadruplets indices, 'target' are dummy labels,
  'preprocessor' are unique points, 'to_transform' are points.
  """
  X, indices = build_data()
  c = np.column_stack(indices)
  target = np.ones(c.shape[0])  # quadruplets targets are not used
  # anyways
  c, target = shuffle(c, target, random_state=SEED)
  if with_preprocessor:
    # if preprocessor, we build a 2D array of quadruplets of indices
    return Dataset(c, target, X, c[:, 0])
  else:
    # if not, we build a 3D array of quadruplets of samples
    return Dataset(X[c], target, None, X[c[:, 0]])


# ------------- List of learners, separating them by kind -------------

# Mahalanobis learners
# -- Weakly Supervised
quadruplets_learners_m = [(LSML(), build_quadruplets)]
ids_quadruplets_learners_m = list(map(lambda x: x.__class__.__name__,
                                      [learner for (learner, _) in
                                       quadruplets_learners_m]))

triplets_learners_m = [(SCML(), build_triplets)]
ids_triplets_learners_m = list(map(lambda x: x.__class__.__name__,
                                   [learner for (learner, _) in
                                    triplets_learners_m]))

pairs_learners_m = [(ITML(max_iter=2), build_pairs),  # max_iter=2 to be faster
                    (MMC(max_iter=2), build_pairs),  # max_iter=2 to be faster
                    (SDML(prior='identity', balance_param=1e-5), build_pairs)]
ids_pairs_learners_m = list(map(lambda x: x.__class__.__name__,
                                [learner for (learner, _) in
                                 pairs_learners_m]))

# -- Supervised
classifiers_m = [(Covariance(), build_classification),
                 (LFDA(), build_classification),
                 (LMNN(), build_classification),
                 (NCA(), build_classification),
                 (RCA(), build_classification),
                 (ITML_Supervised(max_iter=5), build_classification),
                 (LSML_Supervised(), build_classification),
                 (MMC_Supervised(max_iter=5), build_classification),
                 (RCA_Supervised(num_chunks=5), build_classification),
                 (SDML_Supervised(prior='identity', balance_param=1e-5),
                 build_classification),
                 (SCML_Supervised(), build_classification)]
ids_classifiers_m = list(map(lambda x: x.__class__.__name__,
                             [learner for (learner, _) in
                              classifiers_m]))

regressors_m = [(MLKR(init='pca'), build_regression)]
ids_regressors_m = list(map(lambda x: x.__class__.__name__,
                            [learner for (learner, _) in regressors_m]))
# -- Mahalanobis sets
tuples_learners_m = pairs_learners_m + triplets_learners_m + \
                  quadruplets_learners_m
ids_tuples_learners_m = ids_pairs_learners_m + ids_triplets_learners_m \
                      + ids_quadruplets_learners_m

supervised_learners_m = classifiers_m + regressors_m
ids_supervised_learners_m = ids_classifiers_m + ids_regressors_m

metric_learners_m = tuples_learners_m + supervised_learners_m
ids_metric_learners_m = ids_tuples_learners_m + ids_supervised_learners_m

# Bilinear learners
# -- Weakly Supervised
quadruplets_learners_b = [(MockQuadrpletsIdentityBilinearLearner(),
                          build_quadruplets)]
ids_quadruplets_learners_b = list(map(lambda x: x.__class__.__name__,
                                  [learner for (learner, _) in
                                   quadruplets_learners_b]))

triplets_learners_b = [(MockTripletsIdentityBilinearLearner(), build_triplets)]
ids_triplets_learners_b = list(map(lambda x: x.__class__.__name__,
                                   [learner for (learner, _) in
                                    triplets_learners_b]))

pairs_learners_b = [(MockPairIdentityBilinearLearner(), build_pairs)]
ids_pairs_learners_b = list(map(lambda x: x.__class__.__name__,
                                [learner for (learner, _) in
                                 pairs_learners_b]))
# -- Supervised
classifiers_b = [(RandomBilinearLearner(), build_classification),
                 (IdentityBilinearLearner(), build_classification)]

ids_classifiers_b = list(map(lambda x: x.__class__.__name__,
                             [learner for (learner, _) in
                              classifiers_b]))
# -- Bilinear sets
tuples_learners_b = pairs_learners_b + triplets_learners_b + \
                  quadruplets_learners_b
ids_tuples_learners_b = ids_pairs_learners_b + ids_triplets_learners_b \
                      + ids_quadruplets_learners_b

supervised_learners_b = classifiers_b
ids_supervised_learners_b = ids_classifiers_b

metric_learners_b = tuples_learners_b + supervised_learners_b
ids_metric_learners_b = ids_tuples_learners_b + ids_supervised_learners_b

# General sets (Mahalanobis + Bilinear)
# -- Weakly Supervised learners individually
pairs_learners = pairs_learners_m + pairs_learners_b
ids_pairs_learners = ids_pairs_learners_m + ids_pairs_learners_b
triplets_learners = triplets_learners_m + triplets_learners_b
ids_triplets_learners = ids_triplets_learners_m + ids_triplets_learners_b
quadruplets_learners = quadruplets_learners_m + quadruplets_learners_b
ids_quadruplets_learners = ids_quadruplets_learners_m + \
                         ids_quadruplets_learners_b

# -- All weakly supervised learners
tuples_learners = tuples_learners_m + tuples_learners_b
ids_tuples_learners = ids_tuples_learners_m + ids_tuples_learners_b

# -- Supervised learners
supervised_learners = supervised_learners_m + supervised_learners_b
ids_supervised_learners = ids_supervised_learners_m + ids_supervised_learners_b

# -- Weakly Supervised + Supervised learners
metric_learners = metric_learners_m + metric_learners_b
ids_metric_learners = ids_metric_learners_m + ids_metric_learners_b

# -- For sklearn pipeline: Pair + Supervised learners
metric_learners_pipeline = pairs_learners_m + pairs_learners_b + \
                         supervised_learners_m + supervised_learners_b
ids_metric_learners_pipeline = ids_pairs_learners_m + ids_pairs_learners_b +\
                             ids_supervised_learners_m + \
                             ids_supervised_learners_b

# Not used
WeaklySupervisedClasses = (_PairsClassifierMixin,
                           _TripletsClassifierMixin,
                           _QuadrupletsClassifierMixin)

# ------------- Useful methods -------------


def remove_y(estimator, X, y):
  """Quadruplets and triplets learners have no y in fit, but to write test for
  all estimators, it is convenient to have this function, that will return X
  and y if the estimator needs a y to fit on, and just X otherwise."""
  no_y_fit = quadruplets_learners + triplets_learners
  if estimator.__class__.__name__ in [e.__class__.__name__
                                      for (e, _) in no_y_fit]:
    return (X,)
  else:
    return (X, y)


def mock_preprocessor(indices):
  """A preprocessor for testing purposes that returns an all ones 3D array
  """
  return np.ones((indices.shape[0], 3))


@pytest.mark.parametrize('type_of_inputs', ['other', 'tuple', 'classics', 2,
                                            int, NCA()])
def test_check_input_invalid_type_of_inputs(type_of_inputs):
  """Tests that an invalid type of inputs in check_inputs raises an error."""
  with pytest.raises(ValueError) as e:
    check_input([[0.2, 2.1], [0.2, .8]], type_of_inputs=type_of_inputs)
  msg = ("Unknown value {} for type_of_inputs. Valid values are "
         "'classic' or 'tuples'.".format(type_of_inputs))
  assert str(e.value) == msg


#  ---------------- test check_input with 'tuples' type_of_input' ------------


def tuples_prep():
  """Basic array for testing when using a preprocessor"""
  tuples = np.array([[1, 2],
                     [2, 3]])
  return tuples


def tuples_no_prep():
  """Basic array for testing when using no preprocessor"""
  tuples = np.array([[[1., 2.3], [2.3, 5.3]],
                     [[2.3, 4.3], [0.2, 0.4]]])
  return tuples


@pytest.mark.parametrize('estimator, expected',
                         [(NCA(), " by NCA"), ('NCA', " by NCA"), (None, "")])
def test_make_context(estimator, expected):
  """test the make_name function"""
  assert make_context(estimator) == expected


@pytest.mark.parametrize('estimator, expected',
                         [(NCA(), "NCA"), ('NCA', "NCA"), (None, None)])
def test_make_name(estimator, expected):
  """test the make_name function"""
  assert make_name(estimator) == expected


@pytest.mark.parametrize('estimator, context',
                         [(NCA(), " by NCA"), ('NCA', " by NCA"), (None, "")])
@pytest.mark.parametrize('load_tuples, preprocessor',
                         [(tuples_prep, mock_preprocessor),
                          (tuples_no_prep, None),
                          (tuples_no_prep, mock_preprocessor)])
def test_check_tuples_invalid_tuple_size(estimator, context, load_tuples,
                                         preprocessor):
  """Checks that the exception are raised if tuple_size is not the one
  expected"""
  tuples = load_tuples()
  preprocessed_tuples = (preprocess_tuples(tuples, preprocessor)
                         if (preprocessor is not None and
                         tuples.ndim == 2) else tuples)
  expected_msg = ("Tuples of 3 element(s) expected{}. Got tuples of 2 "
                  "element(s) instead (shape={}):\ninput={}.\n"
                  .format(context, preprocessed_tuples.shape,
                          preprocessed_tuples))
  with pytest.raises(ValueError) as raised_error:
    check_input(tuples, type_of_inputs='tuples', tuple_size=3,
                preprocessor=preprocessor, estimator=estimator)
  assert str(raised_error.value) == expected_msg


@pytest.mark.parametrize('estimator, context',
                         [(NCA(), " by NCA"), ('NCA', " by NCA"), (None, "")])
@pytest.mark.parametrize('tuples, found, expected, preprocessor',
                         [(5, '0', '2D array of indicators or 3D array of '
                                   'formed tuples', mock_preprocessor),
                          (5, '0', '3D array of formed tuples', None),
                          ([1, 2], '1', '2D array of indicators or 3D array '
                                        'of formed tuples', mock_preprocessor),
                          ([1, 2], '1', '3D array of formed tuples', None),
                          ([[[[5]]]], '4', '2D array of indicators or 3D array'
                                           ' of formed tuples',
                           mock_preprocessor),
                          ([[[[5]]]], '4', '3D array of formed tuples', None),
                          ([[1], [3]], '2', '3D array of formed '
                                            'tuples', None)])
def test_check_tuples_invalid_shape(estimator, context, tuples, found,
                                    expected, preprocessor):
  """Checks that a value error with the appropriate message is raised if
  shape is invalid (not 2D with preprocessor or 3D with no preprocessor)
  """
  tuples = np.array(tuples)
  msg = ("{} expected{}{}. Found {}D array instead:\ninput={}. Reshape your "
         "data{}.\n"
         .format(expected, context, ' when using a preprocessor'
                 if preprocessor else '', found, tuples,
                 ' and/or use a preprocessor' if
                 (not preprocessor and tuples.ndim == 2) else ''))
  with pytest.raises(ValueError) as raised_error:
      check_input(tuples, type_of_inputs='tuples',
                  preprocessor=preprocessor, ensure_min_samples=0,
                  estimator=estimator)
  assert str(raised_error.value) == msg


@pytest.mark.parametrize('estimator, context',
                         [(NCA(), " by NCA"), ('NCA', " by NCA"), (None, "")])
def test_check_tuples_invalid_n_features(estimator, context):
  """Checks that the right warning is printed if not enough features
  Here we only test if no preprocessor (otherwise we don't ensure this)
  """
  msg = ("Found array with 2 feature(s) (shape={}) while"
         " a minimum of 3 is required{}.".format(tuples_no_prep().shape,
                                                 context))
  with pytest.raises(ValueError) as raised_error:
      check_input(tuples_no_prep(), type_of_inputs='tuples',
                  preprocessor=None, ensure_min_features=3,
                  estimator=estimator)
  assert str(raised_error.value) == msg


@pytest.mark.parametrize('estimator, context',
                         [(NCA(), " by NCA"), ('NCA', " by NCA"), (None, "")])
@pytest.mark.parametrize('load_tuples, preprocessor',
                         [(tuples_prep, mock_preprocessor),
                          (tuples_no_prep, None),
                          (tuples_no_prep, mock_preprocessor)])
def test_check_tuples_invalid_n_samples(estimator, context, load_tuples,
                                        preprocessor):
  """Checks that the right warning is printed if n_samples is too small"""
  tuples = load_tuples()
  msg = ("Found array with 2 sample(s) (shape={}) while a minimum of 3 "
         "is required{}.".format((preprocess_tuples(tuples, preprocessor)
                                 if (preprocessor is not None and
                                 tuples.ndim == 2) else tuples).shape,
                                 context))
  with pytest.raises(ValueError) as raised_error:
    check_input(tuples, type_of_inputs='tuples',
                preprocessor=preprocessor,
                ensure_min_samples=3, estimator=estimator)
  assert str(raised_error.value) == msg


def test_check_tuples_invalid_dtype_not_convertible_with_preprocessor():
  """Checks that a value error is thrown if attempting to convert an
  input not convertible to float, when using a preprocessor
  """

  def preprocessor(indices):
    # preprocessor that returns objects
    return np.full((indices.shape[0], 3), 'a')

  with pytest.raises(ValueError):
    check_input(tuples_prep(), type_of_inputs='tuples',
                preprocessor=preprocessor, dtype=np.float64)


def test_check_tuples_invalid_dtype_not_convertible_without_preprocessor():
  """Checks that a value error is thrown if attempting to convert an
  input not convertible to float, when using no preprocessor
  """
  tuples = np.full_like(tuples_no_prep(), 'a', dtype=object)
  with pytest.raises(ValueError):
    check_input(tuples, type_of_inputs='tuples',
                preprocessor=None, dtype=np.float64)


@pytest.mark.parametrize('tuple_size', [2, None])
def test_check_tuples_valid_tuple_size(tuple_size):
  """For inputs that have the right matrix dimension (2D or 3D for instance),
  checks that checking the number of tuples (pairs, quadruplets, etc) raises
  no warning if there is the right number of points in a tuple.
  """
  with pytest.warns(None) as record:
    check_input(tuples_prep(), type_of_inputs='tuples',
                preprocessor=mock_preprocessor, tuple_size=tuple_size)
    check_input(tuples_no_prep(), type_of_inputs='tuples', preprocessor=None,
                tuple_size=tuple_size)
  assert len(record) == 0


@pytest.mark.parametrize('tuples',
                         [np.array([[2.5, 0.1, 2.6],
                                    [1.6, 4.8, 9.1]]),
                          np.array([[2, 0, 2],
                                    [1, 4, 9]]),
                          np.array([["img1.png", "img3.png"],
                                    ["img2.png", "img4.png"]]),
                          [[2, 0, 2],
                           [1, 4, 9]],
                          [np.array([2, 0, 2]),
                           np.array([1, 4, 9])],
                          ((2, 0, 2),
                           (1, 4, 9)),
                          np.array([[[1.2, 2.2], [1.4, 3.3]],
                                    [[2.6, 2.3], [3.4, 5.0]]])])
def test_check_tuples_valid_with_preprocessor(tuples):
  """Test that valid inputs when using a preprocessor raises no warning"""
  with pytest.warns(None) as record:
    check_input(tuples, type_of_inputs='tuples',
                preprocessor=mock_preprocessor)
  assert len(record) == 0


@pytest.mark.parametrize('tuples',
                         [np.array([[[2.5], [0.1], [2.6]],
                                    [[1.6], [4.8], [9.1]],
                                    [[5.6], [2.8], [6.1]]]),
                          np.array([[[2], [0], [2]],
                                    [[1], [4], [9]],
                                    [[1], [5], [3]]]),
                          [[[2], [0], [2]],
                           [[1], [4], [9]],
                           [[3], [4], [29]]],
                          (((2, 1), (0, 2), (2, 3)),
                           ((1, 2), (4, 4), (9, 3)),
                           ((3, 1), (4, 4), (29, 4)))])
def test_check_tuples_valid_without_preprocessor(tuples):
  """Test that valid inputs when using no preprocessor raises no warning"""
  with pytest.warns(None) as record:
    check_input(tuples, type_of_inputs='tuples', preprocessor=None)
  assert len(record) == 0


def test_check_tuples_behaviour_auto_dtype():
  """Checks that check_tuples allows by default every type if using a
  preprocessor, and numeric types if using no preprocessor"""
  tuples_prep = [['img1.png', 'img2.png'], ['img3.png', 'img5.png']]
  with pytest.warns(None) as record:
    check_input(tuples_prep, type_of_inputs='tuples',
                preprocessor=mock_preprocessor)
  assert len(record) == 0

  with pytest.warns(None) as record:
      check_input(tuples_no_prep(), type_of_inputs='tuples')  # numeric type
  assert len(record) == 0

  # not numeric type
  tuples_no_prep_bis = np.array([[['img1.png'], ['img2.png']],
                                 [['img3.png'], ['img5.png']]])
  tuples_no_prep_bis = tuples_no_prep_bis.astype(object)
  with pytest.raises(ValueError):
      check_input(tuples_no_prep_bis, type_of_inputs='tuples')


def test_check_tuples_invalid_complex_data():
  """Checks that the right error message is thrown if given complex data (
  this comes from sklearn's check_array's message)"""
  tuples = np.array([[[1 + 2j, 3 + 4j], [5 + 7j, 5 + 7j]],
                     [[1 + 3j, 2 + 4j], [5 + 8j, 1 + 7j]]])
  msg = ("Complex data not supported\n"
         "{}\n".format(tuples))
  with pytest.raises(ValueError) as raised_error:
    check_input(tuples, type_of_inputs='tuples')
  assert str(raised_error.value) == msg


# ------------- test check_input with 'classic' type_of_inputs ----------------


def points_prep():
  """Basic array for testing when using a preprocessor"""
  points = np.array([1, 2])
  return points


def points_no_prep():
  """Basic array for testing when using no preprocessor"""
  points = np.array([[1., 2.3],
                     [2.3, 4.3]])
  return points


@pytest.mark.parametrize('estimator, context',
                         [(NCA(), " by NCA"), ('NCA', " by NCA"), (None, "")])
@pytest.mark.parametrize('points, found, expected, preprocessor',
                         [(5, '0', '1D array of indicators or 2D array of '
                                   'formed points', mock_preprocessor),
                          (5, '0', '2D array of formed points', None),
                          ([1, 2], '1', '2D array of formed points', None),
                          ([[[5]]], '3', '1D array of indicators or 2D '
                                         'array of formed points',
                           mock_preprocessor),
                          ([[[5]]], '3', '2D array of formed points', None)])
def test_check_classic_invalid_shape(estimator, context, points, found,
                                     expected, preprocessor):
  """Checks that a value error with the appropriate message is raised if
  shape is invalid (valid being 1D or 2D with preprocessor or 2D with no
  preprocessor)
  """
  points = np.array(points)
  msg = ("{} expected{}{}. Found {}D array instead:\ninput={}. Reshape your "
         "data{}.\n"
         .format(expected, context, ' when using a preprocessor'
                 if preprocessor else '', found, points,
                 ' and/or use a preprocessor' if
                 (not preprocessor and points.ndim == 1) else ''))
  with pytest.raises(ValueError) as raised_error:
    check_input(points, type_of_inputs='classic', preprocessor=preprocessor,
                ensure_min_samples=0,
                estimator=estimator)
  assert str(raised_error.value) == msg


@pytest.mark.parametrize('estimator, context',
                         [(NCA(), " by NCA"), ('NCA', " by NCA"), (None, "")])
def test_check_classic_invalid_n_features(estimator, context):
  """Checks that the right warning is printed if not enough features
  Here we only test if no preprocessor (otherwise we don't ensure this)
  """
  msg = ("Found array with 2 feature(s) (shape={}) while"
         " a minimum of 3 is required{}.".format(points_no_prep().shape,
                                                 context))
  with pytest.raises(ValueError) as raised_error:
      check_input(points_no_prep(), type_of_inputs='classic',
                  preprocessor=None, ensure_min_features=3,
                  estimator=estimator)
  assert str(raised_error.value) == msg


@pytest.mark.parametrize('estimator, context',
                         [(NCA(), " by NCA"), ('NCA', " by NCA"), (None, "")])
@pytest.mark.parametrize('load_points, preprocessor',
                         [(points_prep, mock_preprocessor),
                          (points_no_prep, None),
                          (points_no_prep, mock_preprocessor)])
def test_check_classic_invalid_n_samples(estimator, context, load_points,
                                         preprocessor):
  """Checks that the right warning is printed if n_samples is too small"""
  points = load_points()
  msg = ("Found array with 2 sample(s) (shape={}) while a minimum of 3 "
         "is required{}.".format((preprocess_points(points,
                                                    preprocessor)
                                 if preprocessor is not None and
                                 points.ndim == 1 else
                                 points).shape,
                                 context))
  with pytest.raises(ValueError) as raised_error:
    check_input(points, type_of_inputs='classic', preprocessor=preprocessor,
                ensure_min_samples=3,
                estimator=estimator)
  assert str(raised_error.value) == msg


@pytest.mark.parametrize('preprocessor, points',
                         [(mock_preprocessor, np.array([['a', 'b'],
                                                        ['e', 'b']])),
                          (None, np.array([[['b', 'v'], ['a', 'd']],
                                           [['x', 'u'], ['c', 'a']]]))])
def test_check_classic_invalid_dtype_not_convertible(preprocessor, points):
  """Checks that a value error is thrown if attempting to convert an
  input not convertible to float
  """
  with pytest.raises(ValueError):
    check_input(points, type_of_inputs='classic',
                preprocessor=preprocessor, dtype=np.float64)


@pytest.mark.parametrize('points',
                         [["img1.png", "img3.png", "img2.png"],
                          np.array(["img1.png", "img3.png", "img2.png"]),
                          [2, 0, 2, 1, 4, 9],
                          range(10),
                          np.array([2, 0, 2]),
                          (2, 0, 2),
                          np.array([[1.2, 2.2],
                                    [2.6, 2.3]])])
def test_check_classic_valid_with_preprocessor(points):
  """Test that valid inputs when using a preprocessor raises no warning"""
  with pytest.warns(None) as record:
    check_input(points, type_of_inputs='classic',
                preprocessor=mock_preprocessor)
  assert len(record) == 0


@pytest.mark.parametrize('points',
                         [np.array([[2.5, 0.1, 2.6],
                                    [1.6, 4.8, 9.1],
                                    [5.6, 2.8, 6.1]]),
                          np.array([[2, 0, 2],
                                    [1, 4, 9],
                                    [1, 5, 3]]),
                          [[2, 0, 2],
                           [1, 4, 9],
                           [3, 4, 29]],
                          ((2, 1, 0, 2, 2, 3),
                           (1, 2, 4, 4, 9, 3),
                           (3, 1, 4, 4, 29, 4))])
def test_check_classic_valid_without_preprocessor(points):
  """Test that valid inputs when using no preprocessor raises no warning"""
  with pytest.warns(None) as record:
    check_input(points, type_of_inputs='classic', preprocessor=None)
  assert len(record) == 0


def test_check_classic_by_default():
  """Checks that 'classic' is the default behaviour of check_input"""
  assert (check_input([[2, 3], [3, 2]]) ==
          check_input([[2, 3], [3, 2]], type_of_inputs='classic')).all()


def test_check_classic_behaviour_auto_dtype():
  """Checks that check_input (for points) allows by default every type if
  using a preprocessor, and numeric types if using no preprocessor"""
  points_prep = ['img1.png', 'img2.png', 'img3.png', 'img5.png']
  with pytest.warns(None) as record:
    check_input(points_prep, type_of_inputs='classic',
                preprocessor=mock_preprocessor)
  assert len(record) == 0

  with pytest.warns(None) as record:
      check_input(points_no_prep(), type_of_inputs='classic')  # numeric type
  assert len(record) == 0

  # not numeric type
  points_no_prep_bis = np.array(['img1.png', 'img2.png', 'img3.png',
                                 'img5.png'])
  points_no_prep_bis = points_no_prep_bis.astype(object)
  with pytest.raises(ValueError):
      check_input(points_no_prep_bis, type_of_inputs='classic')


def test_check_classic_invalid_complex_data():
  """Checks that the right error message is thrown if given complex data (
  this comes from sklearn's check_array's message)"""
  points = np.array([[[1 + 2j, 3 + 4j], [5 + 7j, 5 + 7j]],
                     [[1 + 3j, 2 + 4j], [5 + 8j, 1 + 7j]]])
  msg = ("Complex data not supported\n"
         "{}\n".format(points))
  with pytest.raises(ValueError) as raised_error:
    check_input(points, type_of_inputs='classic')
  assert str(raised_error.value) == msg


# ----------------------------- Test preprocessor -----------------------------


X = np.array([[0.89, 0.11, 1.48, 0.12],
              [2.63, 1.08, 1.68, 0.46],
              [1.00, 0.59, 0.62, 1.15]])


class MockFileLoader:
  """Preprocessor that takes a root file path at construction and simulates
  fetching the file in the specific root folder when given the name of the
  file"""

  def __init__(self, root):
    self.root = root
    self.folders = {'fake_root': {'img0.png': X[0],
                                  'img1.png': X[1],
                                  'img2.png': X[2]
                                  },
                    'other_folder': {}  # empty folder
                    }

  def __call__(self, path_list):
    images = list()
    for path in path_list:
      images.append(self.folders[self.root][path])
    return np.array(images)


def mock_id_loader(list_of_indicators):
  """A preprocessor as a function that takes indicators (strings) and
  returns the corresponding samples"""
  points = []
  for indicator in list_of_indicators:
      points.append(X[int(indicator[2:])])
  return np.array(points)


tuples_list = [np.array([[0, 1],
                         [2, 1]]),

               np.array([['img0.png', 'img1.png'],
                         ['img2.png', 'img1.png']]),

               np.array([['id0', 'id1'],
                         ['id2', 'id1']])
               ]

points_list = [np.array([0, 1, 2, 1]),

               np.array(['img0.png', 'img1.png', 'img2.png', 'img1.png']),

               np.array(['id0', 'id1', 'id2', 'id1'])
               ]

preprocessors = [X, MockFileLoader('fake_root'), mock_id_loader]


@pytest.fixture
def y_tuples():
  y = [-1, 1]
  return y


@pytest.fixture
def y_points():
  y = [0, 1, 0, 0]
  return y


@pytest.mark.parametrize('preprocessor, tuples', zip(preprocessors,
                                                     tuples_list))
def test_preprocessor_weakly_supervised(preprocessor, tuples, y_tuples):
  """Tests different ways to use the preprocessor argument: an array,
  a class callable, and a function callable, with a weakly supervised
  algorithm
  """
  nca = ITML(preprocessor=preprocessor)
  nca.fit(tuples, y_tuples)


@pytest.mark.parametrize('preprocessor, points', zip(preprocessors,
                                                     points_list))
def test_preprocessor_supervised(preprocessor, points, y_points):
  """Tests different ways to use the preprocessor argument: an array,
  a class callable, and a function callable, with a supervised algorithm
  """
  lfda = LFDA(preprocessor=preprocessor)
  lfda.fit(points, y_points)


@pytest.mark.parametrize('estimator', ['NCA', NCA(), None])
def test_preprocess_tuples_invalid_message(estimator):
  """Checks that if the preprocessor does some weird stuff, the preprocessed
  input is detected as weird. Checks this for preprocess_tuples."""

  context = make_context(estimator) + (' after the preprocessor '
                                       'has been applied')

  def preprocessor(sequence):
    return np.ones((len(sequence), 2, 2))  # returns a 3D array instead of 2D

  with pytest.raises(ValueError) as raised_error:
      check_input(np.ones((3, 2)), type_of_inputs='tuples',
                  preprocessor=preprocessor, estimator=estimator)
  expected_msg = ("3D array of formed tuples expected{}. Found 4D "
                  "array instead:\ninput={}. Reshape your data{}.\n"
                  .format(context, np.ones((3, 2, 2, 2)),
                          ' and/or use a preprocessor' if preprocessor
                          is not None else ''))
  assert str(raised_error.value) == expected_msg


@pytest.mark.parametrize('estimator', ['NCA', NCA(), None])
def test_preprocess_points_invalid_message(estimator):
  """Checks that if the preprocessor does some weird stuff, the preprocessed
  input is detected as weird."""

  context = make_context(estimator) + (' after the preprocessor '
                                       'has been applied')

  def preprocessor(sequence):
    return np.ones((len(sequence), 2, 2))  # returns a 3D array instead of 2D

  with pytest.raises(ValueError) as raised_error:
    check_input(np.ones((3,)), type_of_inputs='classic',
                preprocessor=preprocessor, estimator=estimator)
  expected_msg = ("2D array of formed points expected{}. "
                  "Found 3D array instead:\ninput={}. Reshape your data{}.\n"
                  .format(context, np.ones((3, 2, 2)),
                          ' and/or use a preprocessor' if preprocessor
                          is not None else ''))
  assert str(raised_error.value) == expected_msg


def test_preprocessor_error_message():
  """Tests whether the preprocessor returns a preprocessor error when there
  is a problem using the preprocessor
  """
  preprocessor = ArrayIndexer(np.array([[1.2, 3.3], [3.1, 3.2]]))

  # with tuples
  X = np.array([[[2, 3], [3, 3]], [[2, 3], [3, 2]]])
  # There are less samples than the max index we want to preprocess
  with pytest.raises(PreprocessorError):
    preprocess_tuples(X, preprocessor)

  # with points
  X = np.array([[1], [2], [3], [3]])
  with pytest.raises(PreprocessorError):
    preprocess_points(X, preprocessor)


@pytest.mark.parametrize('input_data', [[[5, 3], [3, 2]],
                                        ((5, 3), (3, 2))
                                        ])
@pytest.mark.parametrize('indices', [[0, 1], (1, 0)])
def test_array_like_indexer_array_like_valid_classic(input_data, indices):
  """Checks that any array-like is valid in the 'preprocessor' argument,
  and in the indices, for a classic input"""
  class MockMetricLearner(MahalanobisMixin):
    def fit(self):
      pass
    pass

  mock_algo = MockMetricLearner(preprocessor=input_data)
  mock_algo._prepare_inputs(indices, type_of_inputs='classic')


@pytest.mark.parametrize('input_data', [[[5, 3], [3, 2]],
                                        ((5, 3), (3, 2))
                                        ])
@pytest.mark.parametrize('indices', [[[0, 1], [1, 0]], ((1, 0), (1, 0))])
def test_array_like_indexer_array_like_valid_tuples(input_data, indices):
  """Checks that any array-like is valid in the 'preprocessor' argument,
  and in the indices, for a classic input"""
  class MockMetricLearner(MahalanobisMixin):
    def fit(self):
      pass
    pass

  mock_algo = MockMetricLearner(preprocessor=input_data)
  mock_algo._prepare_inputs(indices, type_of_inputs='tuples')


@pytest.mark.parametrize('preprocessor', [4, NCA()])
def test_error_message_check_preprocessor(preprocessor):
  """Checks that if the preprocessor given is not an array-like or a
  callable, the right error message is returned"""
  class MockMetricLearner(MahalanobisMixin):
    pass

  mock_algo = MockMetricLearner(preprocessor=preprocessor)
  with pytest.raises(ValueError) as e:
    mock_algo._check_preprocessor()
  assert str(e.value) == ("Invalid type for the preprocessor: {}. You should "
                          "provide either None, an array-like object, "
                          "or a callable.".format(type(preprocessor)))


@pytest.mark.parametrize('estimator, _', tuples_learners,
                         ids=ids_tuples_learners)
def test_error_message_tuple_size(estimator, _):
  """Tests that if a tuples learner is not given the good number of points
  per tuple, it throws an error message"""
  estimator = clone(estimator)
  set_random_state(estimator)
  invalid_pairs = np.ones((2, 5, 2))
  y = [1, 1]
  with pytest.raises(ValueError) as raised_err:
    estimator.fit(*remove_y(estimator, invalid_pairs, y))
  expected_msg = ("Tuples of {} element(s) expected{}. Got tuples of 5 "
                  "element(s) instead (shape=(2, 5, 2)):\ninput={}.\n"
                  .format(estimator._tuple_size, make_context(estimator),
                          invalid_pairs))
  assert str(raised_err.value) == expected_msg


@pytest.mark.parametrize('estimator, _', metric_learners,
                         ids=ids_metric_learners)
def test_error_message_t_pair_distance_or_score(estimator, _):
  """Tests that if you want to pair_distance or pair_score on triplets
  for instance, it returns the right error message
  """
  estimator = clone(estimator)
  set_random_state(estimator)
  estimator._check_preprocessor()
  triplets = np.array([[[1.3, 6.3], [3., 6.8], [6.5, 4.4]],
                       [[1.9, 5.3], [1., 7.8], [3.2, 1.2]]])
  with pytest.raises(ValueError) as raised_err:
    estimator.pair_score(triplets)
  expected_msg = ("Tuples of 2 element(s) expected{}. Got tuples of 3 "
                  "element(s) instead (shape=(2, 3, 2)):\ninput={}.\n"
                  .format(make_context(estimator), triplets))
  assert str(raised_err.value) == expected_msg

  msg = ("This learner doesn't learn a distance, thus ",
         "this method is not implemented. Use pair_score instead")

  # One exception will trigger for sure
  with pytest.raises(Exception) as raised_exception:
      estimator.pair_distance(triplets)
  err_value = raised_exception.value.args[0]
  assert err_value == expected_msg or err_value == msg


def test_preprocess_tuples_simple_example():
  """Test the preprocessor on a very simple example of tuples to ensure the
  result is as expected"""
  array = np.array([[1, 2],
                    [2, 3],
                    [4, 5]])

  def fun(row):
    return np.array([[1, 1], [3, 3], [4, 4]])

  expected_result = np.array([[[1, 1], [1, 1]],
                              [[3, 3], [3, 3]],
                              [[4, 4], [4, 4]]])

  assert (preprocess_tuples(array, fun) == expected_result).all()


def test_preprocess_points_simple_example():
  """Test the preprocessor on very simple examples of points to ensure the
  result is as expected"""
  array = np.array([1, 2, 4])

  def fun(row):
    return [[1, 1], [3, 3], [4, 4]]

  expected_result = np.array([[1, 1],
                              [3, 3],
                              [4, 4]])

  assert (preprocess_points(array, fun) == expected_result).all()


# TODO: Find a better way to run this test and the next one, to avoid
# duplicated code.
@pytest.mark.parametrize('estimator, build_dataset', metric_learners_m,
                         ids=ids_metric_learners_m)
def test_same_with_or_without_preprocessor_mahalanobis(estimator,
                                                       build_dataset):
  """Test that Mahalanobis algorithms using a preprocessor behave
  consistently with their no-preprocessor equivalent. Methods
  `pair_distance` and `transform`.
  """
  dataset_indices = build_dataset(with_preprocessor=True)
  dataset_formed = build_dataset(with_preprocessor=False)
  X = dataset_indices.preprocessor
  indicators_to_transform = dataset_indices.to_transform
  formed_points_to_transform = dataset_formed.to_transform
  (indices_train, indices_test, y_train, y_test, formed_train,
   formed_test) = train_test_split(dataset_indices.data,
                                   dataset_indices.target,
                                   dataset_formed.data,
                                   random_state=SEED)
  estimator_with_preprocessor = clone(estimator)
  set_random_state(estimator_with_preprocessor)
  estimator_with_preprocessor.set_params(preprocessor=X)
  estimator_with_preprocessor.fit(*remove_y(estimator, indices_train, y_train))

  estimator_without_preprocessor = clone(estimator)
  set_random_state(estimator_without_preprocessor)
  estimator_without_preprocessor.set_params(preprocessor=None)
  estimator_without_preprocessor.fit(*remove_y(estimator, formed_train,
                                               y_train))
  estimator_with_prep_formed = clone(estimator)
  set_random_state(estimator_with_prep_formed)
  estimator_with_prep_formed.set_params(preprocessor=X)
  estimator_with_prep_formed.fit(*remove_y(estimator, indices_train, y_train))
  idx1 = np.array([[0, 2], [5, 3]], dtype=int)  # Sample

  # Pair distance
  output_with_prep = estimator_with_preprocessor.pair_distance(
      indicators_to_transform[idx1])
  output_without_prep = estimator_without_preprocessor.pair_distance(
      formed_points_to_transform[idx1])
  assert np.array(output_with_prep == output_without_prep).all()

  output_with_prep = estimator_with_preprocessor.pair_distance(
      indicators_to_transform[idx1])
  output_without_prep = estimator_with_prep_formed.pair_distance(
      formed_points_to_transform[idx1])
  assert np.array(output_with_prep == output_without_prep).all()

  # Transform
  output_with_prep = estimator_with_preprocessor.transform(
      indicators_to_transform)
  output_without_prep = estimator_without_preprocessor.transform(
      formed_points_to_transform)
  assert np.array(output_with_prep == output_without_prep).all()

  output_with_prep = estimator_with_preprocessor.transform(
      indicators_to_transform)
  output_without_prep = estimator_with_prep_formed.transform(
      formed_points_to_transform)
  assert np.array(output_with_prep == output_without_prep).all()


@pytest.mark.parametrize('estimator, build_dataset', metric_learners,
                         ids=ids_metric_learners)
def test_same_with_or_without_preprocessor(estimator, build_dataset):
  """Test that algorithms using a preprocessor behave consistently
  with their no-preprocessor equivalent. Methods `pair_score`,
  `score_pairs` (deprecated), `predict` and `decision_function`.
  """
  dataset_indices = build_dataset(with_preprocessor=True)
  dataset_formed = build_dataset(with_preprocessor=False)
  X = dataset_indices.preprocessor
  indicators_to_transform = dataset_indices.to_transform
  formed_points_to_transform = dataset_formed.to_transform
  (indices_train, indices_test, y_train, y_test, formed_train,
   formed_test) = train_test_split(dataset_indices.data,
                                   dataset_indices.target,
                                   dataset_formed.data,
                                   random_state=SEED)

  estimator_with_preprocessor = clone(estimator)
  set_random_state(estimator_with_preprocessor)
  estimator_with_preprocessor.set_params(preprocessor=X)
  estimator_with_preprocessor.fit(*remove_y(estimator, indices_train, y_train))

  estimator_without_preprocessor = clone(estimator)
  set_random_state(estimator_without_preprocessor)
  estimator_without_preprocessor.set_params(preprocessor=None)
  estimator_without_preprocessor.fit(*remove_y(estimator, formed_train,
                                               y_train))

  estimator_with_prep_formed = clone(estimator)
  set_random_state(estimator_with_prep_formed)
  estimator_with_prep_formed.set_params(preprocessor=X)
  estimator_with_prep_formed.fit(*remove_y(estimator, indices_train, y_train))

  # Test prediction methods for Weakly supervised algorithms.
  for method in ["predict", "decision_function"]:
    if hasattr(estimator, method):
      output_with_prep = getattr(estimator_with_preprocessor,
                                 method)(indices_test)
      output_without_prep = getattr(estimator_without_preprocessor,
                                    method)(formed_test)
      assert np.array(output_with_prep == output_without_prep).all()
      output_with_prep = getattr(estimator_with_preprocessor,
                                 method)(indices_test)
      output_with_prep_formed = getattr(estimator_with_prep_formed,
                                        method)(formed_test)
      assert np.array(output_with_prep == output_with_prep_formed).all()

  idx1 = np.array([[0, 2], [5, 3]], dtype=int)  # Sample

  # Pair score
  output_with_prep = estimator_with_preprocessor.pair_score(
      indicators_to_transform[idx1])
  output_without_prep = estimator_without_preprocessor.pair_score(
      formed_points_to_transform[idx1])
  assert np.array(output_with_prep == output_without_prep).all()

  output_with_prep = estimator_with_preprocessor.pair_score(
      indicators_to_transform[idx1])
  output_without_prep = estimator_with_prep_formed.pair_score(
      formed_points_to_transform[idx1])
  assert np.array(output_with_prep == output_without_prep).all()

  # Score pairs. TODO: Delete in 0.8.0
  msg = ("score_pairs will be deprecated in release 0.7.0. "
         "Use pair_score to compute similarity scores, or "
         "pair_distances to compute distances.")
  with pytest.warns(FutureWarning) as raised_warning:
    output_with_prep = estimator_with_preprocessor.score_pairs(
        indicators_to_transform[idx1])
    output_without_prep = estimator_without_preprocessor.score_pairs(
        formed_points_to_transform[idx1])
    assert np.array(output_with_prep == output_without_prep).all()

    output_with_prep = estimator_with_preprocessor.score_pairs(
        indicators_to_transform[idx1])
    output_without_prep = estimator_with_prep_formed.score_pairs(
        formed_points_to_transform[idx1])
    assert np.array(output_with_prep == output_without_prep).all()
  assert any([str(warning.message) == msg for warning in raised_warning])


def test_check_collapsed_pairs_raises_no_error():
  """Checks that check_collapsed_pairs raises no error if no collapsed pairs
  is present"""
  pairs_ok = np.array([[[0.1, 3.3], [3.3, 0.1]],
                       [[0.1, 3.3], [3.3, 0.1]],
                       [[2.5, 8.1], [0.1, 3.3]]])
  check_collapsed_pairs(pairs_ok)


def test_check_collapsed_pairs_raises_error():
  """Checks that check_collapsed_pairs raises no error if no collapsed pairs
  is present"""
  pairs_not_ok = np.array([[[0.1, 3.3], [0.1, 3.3]],
                           [[0.1, 3.3], [3.3, 0.1]],
                           [[2.5, 8.1], [2.5, 8.1]]])
  with pytest.raises(ValueError) as e:
    check_collapsed_pairs(pairs_not_ok)
  assert str(e.value) == ("2 collapsed pairs found (where the left element is "
                          "the same as the right element), out of 3 pairs in"
                          " total.")


def test__validate_vector():
  """Replica of scipy.spatial.tests.test_distance.test__validate_vector"""
  x = [1, 2, 3]
  y = validate_vector(x)
  assert_array_equal(y, x)

  y = validate_vector(x, dtype=np.float64)
  assert_array_equal(y, x)
  assert_equal(y.dtype, np.float64)

  x = [1]
  y = validate_vector(x)
  assert_equal(y.ndim, 1)
  assert_equal(y, x)

  x = 1
  y = validate_vector(x)
  assert_equal(y.ndim, 1)
  assert_equal(y, [x])

  x = np.arange(5).reshape(1, -1, 1)
  y = validate_vector(x)
  assert_equal(y.ndim, 1)
  assert_array_equal(y, x[0, :, 0])

  x = [[1, 2], [3, 4]]
  with pytest.raises(ValueError):
    validate_vector(x)


def test__check_sdp_from_eigen_positive_err_messages():
  """Tests that if _check_sdp_from_eigen is given a negative tol it returns
  an error, and if positive (or None) it does not"""
  w = np.abs(np.random.RandomState(42).randn(10)) + 1
  with pytest.raises(ValueError) as raised_error:
    _check_sdp_from_eigen(w, -5.)
  assert str(raised_error.value) == "tol should be positive."
  with pytest.raises(ValueError) as raised_error:
    _check_sdp_from_eigen(w, -1e-10)
  assert str(raised_error.value) == "tol should be positive."
  _check_sdp_from_eigen(w, 1.)
  _check_sdp_from_eigen(w, 0.)
  _check_sdp_from_eigen(w, None)


@pytest.mark.unit
@pytest.mark.parametrize('w', [np.array([-1.2, 5.5, 6.6]),
                               np.array([-1.2, -5.6])])
def test__check_sdp_from_eigen_positive_eigenvalues(w):
  """Tests that _check_sdp_from_eigen, returns a NonPSDError when
  the eigenvalues are negatives or null."""
  with pytest.raises(NonPSDError):
    _check_sdp_from_eigen(w)


@pytest.mark.unit
@pytest.mark.parametrize('w', [np.array([0., 2.3, 5.3]),
                               np.array([1e-20, 3.5]),
                               np.array([1.5, 2.4, 4.6])])
def test__check_sdp_from_eigen_negative_eigenvalues(w):
  """Tests that _check_sdp_from_eigen, returns no error when the
  eigenvalues are positive."""
  _check_sdp_from_eigen(w)


@pytest.mark.unit
@pytest.mark.parametrize('w, is_definite', [(np.array([1e-15, 5.6]), False),
                                            (np.array([-1e-15, 5.6]), False),
                                            (np.array([3.2, 5.6, 0.01]), True),
                                            ])
def test__check_sdp_from_eigen_returns_definiteness(w, is_definite):
  """Tests that _check_sdp_from_eigen returns the definiteness of the
  matrix (when it is PSD), based on the given eigenvalues"""
  assert _check_sdp_from_eigen(w) == is_definite


@pytest.mark.unit
@pytest.mark.parametrize('w, tol, is_definite',
                         [(np.array([5., 3.]), 2, True),
                          (np.array([5., 1.]), 2, False),
                          (np.array([5., -1.]), 2, False)])
def test__check_sdp_from_eigen_tol_psd(w, tol, is_definite):
  """Tests that _check_sdp_from_eigen, for PSD matrices, returns
  False if an eigenvalue is lower than tol"""
  assert _check_sdp_from_eigen(w, tol=tol) == is_definite


@pytest.mark.unit
@pytest.mark.parametrize('w, tol',
                         [(np.array([5., -3.]), 2),
                          (np.array([1., -3.]), 2)])
def test__check_sdp_from_eigen_tol_non_psd(w, tol):
  """Tests that _check_sdp_from_eigen raises a NonPSDError
  when there is a negative value with abs value higher than tol"""
  with pytest.raises(NonPSDError):
    _check_sdp_from_eigen(w, tol=tol)


@pytest.mark.unit
@pytest.mark.parametrize('w, is_definite',
                         [(np.array([1e5, 1e5, 1e5, 1e5,
                                     1e5, 1e5, 1e-20]), False),
                          (np.array([1e-10, 1e-10]), True)])
def test__check_sdp_from_eigen_tol_default_psd(w, is_definite):
  """Tests that the default tol argument gives good results for edge cases
  like even if the determinant is high but clearly one eigenvalue is low,
  (undefinite so returns False) or when all eigenvalues are low (definite so
  returns True)"""
  assert _check_sdp_from_eigen(w, tol=None) == is_definite


@pytest.mark.unit
@pytest.mark.parametrize('w',
                         [np.array([1., -1.]),
                          np.array([-1e-10, 1e-10])])
def test__check_sdp_from_eigen_tol_default_non_psd(w):
  """Tests that the default tol argument is good for raising
  NonPSDError, e.g. that when a value is clearly relatively
  negative it raises such an error"""
  with pytest.raises(NonPSDError):
    _check_sdp_from_eigen(w, tol=None)


def test__check_n_components():
  """Checks that n_components returns what is expected
  (including the errors)"""
  dim = _check_n_components(5, None)
  assert dim == 5

  dim = _check_n_components(5, 3)
  assert dim == 3

  with pytest.raises(ValueError) as expected_err:
    _check_n_components(5, 10)
  assert str(expected_err.value) == 'Invalid n_components, must be in [1, 5]'

  with pytest.raises(ValueError) as expected_err:
    _check_n_components(5, 0)
  assert str(expected_err.value) == 'Invalid n_components, must be in [1, 5]'


@pytest.mark.unit
@pytest.mark.parametrize('wrong_labels',
                         [[0.5, 0.6, 0.7, 0.8, 0.9],
                          np.random.RandomState(42).randn(5),
                          np.random.RandomState(42).choice([0, 1], size=5)])
def test_check_y_valid_values_for_pairs(wrong_labels):
  expected_msg = ("When training on pairs, the labels (y) should contain "
                  "only values in [-1, 1]. Found an incorrect value.")
  with pytest.raises(ValueError) as raised_error:
    check_y_valid_values_for_pairs(wrong_labels)
  assert str(raised_error.value) == expected_msg


@pytest.mark.integration
@pytest.mark.parametrize('wrong_labels',
                         [[0.5, 0.6, 0.7, 0.8, 0.9],
                          np.random.RandomState(42).randn(5),
                          np.random.RandomState(42).choice([0, 1], size=5)])
def test_check_input_invalid_tuples_without_preprocessor(wrong_labels):
  pairs = np.random.RandomState(42).randn(5, 2, 3)
  expected_msg = ("When training on pairs, the labels (y) should contain "
                  "only values in [-1, 1]. Found an incorrect value.")
  with pytest.raises(ValueError) as raised_error:
    check_input(pairs, wrong_labels, preprocessor=None,
                type_of_inputs='tuples')
  assert str(raised_error.value) == expected_msg


@pytest.mark.integration
@pytest.mark.parametrize('wrong_labels',
                         [[0.5, 0.6, 0.7, 0.8, 0.9],
                          np.random.RandomState(42).randn(5),
                          np.random.RandomState(42).choice([0, 1], size=5)])
def test_check_input_invalid_tuples_with_preprocessor(wrong_labels):
  n_samples, n_features, n_pairs = 10, 4, 5
  rng = np.random.RandomState(42)
  pairs = rng.randint(10, size=(n_pairs, 2))
  preprocessor = rng.randn(n_samples, n_features)
  expected_msg = ("When training on pairs, the labels (y) should contain "
                  "only values in [-1, 1]. Found an incorrect value.")
  with pytest.raises(ValueError) as raised_error:
    check_input(pairs, wrong_labels, preprocessor=ArrayIndexer(preprocessor),
                type_of_inputs='tuples')
  assert str(raised_error.value) == expected_msg


@pytest.mark.integration
@pytest.mark.parametrize('with_preprocessor', [True, False])
@pytest.mark.parametrize('estimator, build_dataset', pairs_learners,
                         ids=ids_pairs_learners)
def test_check_input_pairs_learners_invalid_y(estimator, build_dataset,
                                              with_preprocessor):
  """checks that the only allowed labels for learning pairs are +1 and -1"""
  input_data, labels, _, X = build_dataset()
  wrong_labels_list = [labels + 0.5,
                       np.random.RandomState(42).randn(len(labels)),
                       np.random.RandomState(42).choice([0, 1],
                                                        size=len(labels))]
  model = clone(estimator)
  set_random_state(model)

  expected_msg = ("When training on pairs, the labels (y) should contain "
                  "only values in [-1, 1]. Found an incorrect value.")

  for wrong_labels in wrong_labels_list:
    with pytest.raises(ValueError) as raised_error:
      model.fit(input_data, wrong_labels)
  assert str(raised_error.value) == expected_msg


@pytest.mark.parametrize('has_classes, n_features, n_samples, n_components, '
                         'n_classes, result',
                         [(False, 3, 20, 3, 0, 'identity'),
                          (False, 3, 2, 3, 0, 'identity'),
                          (False, 5, 3, 4, 0, 'identity'),
                          (False, 4, 5, 3, 0, 'pca'),
                          (True, 5, 6, 3, 4, 'lda'),
                          (True, 6, 3, 3, 3, 'identity'),
                          (True, 5, 6, 4, 2, 'pca'),
                          (True, 2, 6, 2, 10, 'lda'),
                          (True, 4, 6, 2, 3, 'lda')
                          ])
def test__auto_select_init(has_classes, n_features, n_samples, n_components,
                           n_classes,
                           result):
  """Checks that the auto selection of the init works as expected"""
  assert (_auto_select_init(has_classes, n_features,
                            n_samples, n_components, n_classes) == result)


@pytest.mark.parametrize('w0', [1e-20, 0., -1e-20])
def test_pseudo_inverse_from_eig_and_pinvh_singular(w0):
  """Checks that _pseudo_inverse_from_eig returns the same result as
  scipy.linalg.pinvh for a singular matrix"""
  rng = np.random.RandomState(SEED)
  A = rng.rand(100, 100)
  A = A + A.T
  w, V = eigh(A)
  w[0] = w0
  A = V.dot(np.diag(w)).dot(V.T)
  np.testing.assert_allclose(_pseudo_inverse_from_eig(w, V), pinvh(A),
                             rtol=1e-05)


def test_pseudo_inverse_from_eig_and_pinvh_nonsingular():
  """Checks that _pseudo_inverse_from_eig returns the same result as
  scipy.linalg.pinvh for a non singular matrix"""
  rng = np.random.RandomState(SEED)
  A = rng.rand(100, 100)
  A = A + A.T
  w, V = eigh(A, check_finite=False)
  np.testing.assert_allclose(_pseudo_inverse_from_eig(w, V), pinvh(A))


@pytest.mark.parametrize(('n_triplets', 'n_iter'),
                         [(10, 10), (33, 70), (100, 67),
                         (10000, 20000)])
def test_indices_funct(n_triplets, n_iter):
  """
  This test verifies the behaviour of _get_random_indices. The
  method used inside OASIS that defines the order in which the
  triplets are given to the algorithm, in an online manner.
  """
  # Not random cases
  base = np.arange(n_triplets)

  # n_iter = n_triplets
  if n_iter == n_triplets:
    r = _get_random_indices(n_triplets=n_triplets, n_iter=n_iter,
                            shuffle=False, random=False,
                            random_state=RNG)
    assert_array_equal(r, base)  # No shuffle
    assert len(r) == len(base)  # Same lenght

    # Shuffle
    r = _get_random_indices(n_triplets=n_triplets, n_iter=n_iter,
                            shuffle=True, random=False,
                            random_state=RNG)
    with assert_raises(AssertionError):  # Should be different
      assert_array_equal(r, base)
    # But contain the same elements
    assert_array_equal(np.unique(r), np.unique(base))
    assert len(r) == len(base)  # Same lenght

  # n_iter > n_triplets
  if n_iter > n_triplets:
    r = _get_random_indices(n_triplets=n_triplets, n_iter=n_iter,
                            shuffle=False, random=False,
                            random_state=RNG)
    assert_array_equal(r[:n_triplets], base)  # First n_triplets must match
    assert len(r) == n_iter  # Expected lenght

    # Next n_iter-n_triplets must be in range(n_triplets)
    sample = r[n_triplets:]
    for i in range(n_iter - n_triplets):
      if sample[i] not in base:
        raise AssertionError("Sampling has values out of range")

    # Shuffle
    r = _get_random_indices(n_triplets=n_triplets, n_iter=n_iter,
                            shuffle=True, random=False,
                            random_state=RNG)
    assert len(r) == n_iter  # Expected lenght

    # Each triplet must be at least one time
    assert_array_equal(np.unique(r), np.unique(base))
    with assert_raises(AssertionError):  # First n_triplets should be different
      assert_array_equal(r[:n_triplets], base)

    # Each index should appear at least ceil(n_iter/n_triplets) - 1 times
    # But no more than ceil(n_iter/n_triplets)
    min_times = int(np.ceil(n_iter / n_triplets)) - 1
    _, counts = np.unique(r, return_counts=True)
    a = len(counts[counts >= min_times])
    b = len(counts[counts <= min_times + 1])
    assert len(np.unique(r)) == a
    assert n_triplets == b

  # n_iter < n_triplets
  if n_iter < n_triplets:
    r = _get_random_indices(n_triplets=n_triplets, n_iter=n_iter,
                            shuffle=False, random=False,
                            random_state=RNG)
    assert len(r) == n_iter  # Expected lenght
    u = np.unique(r)
    assert len(u) == len(r)  # No duplicates
    # Final array must cointain only elements in range(n_triplets)
    for i in range(n_iter):
      if r[i] not in base:
        raise AssertionError("Sampling has values out of range")

    # Shuffle must only sort elements
    # It takes two instances with same random_state, to show that only
    # the final order is mixed
    def is_sorted(a):
      return np.all(a[:-1] <= a[1:])

    r_a = _get_random_indices(n_triplets=n_triplets, n_iter=n_iter,
                              shuffle=False, random=False,
                              random_state=SEED)
    assert is_sorted(r_a)  # Its not shuffled
    values_r_a, counts_r_a = np.unique(r_a, return_counts=True)

    r_b = _get_random_indices(n_triplets=n_triplets, n_iter=n_iter,
                              shuffle=True, random=False,
                              random_state=SEED)

    with assert_raises(AssertionError):
      assert is_sorted(r_b)  # This one should not besorted, but shuffled
    values_r_b, counts_r_b = np.unique(r_b, return_counts=True)

    assert_array_equal(values_r_a, values_r_b)  # Same elements
    assert_array_equal(counts_r_a, counts_r_b)  # Same counts
    with assert_raises(AssertionError):
      assert_array_equal(r_a, r_b)  # Diferent order

  # Random case
  r = _get_random_indices(n_triplets=n_triplets, n_iter=n_iter,
                          random=True, random_state=RNG)
  assert len(r) == n_iter  # Expected lenght
  for i in range(n_iter):
    if r[i] not in base:
      raise AssertionError("Sampling has values out of range")
  # Shuffle has no effect
  r_a = _get_random_indices(n_triplets=n_triplets, n_iter=n_iter,
                            shuffle=False, random=True,
                            random_state=SEED)

  r_b = _get_random_indices(n_triplets=n_triplets, n_iter=n_iter,
                            shuffle=True, random=True,
                            random_state=SEED)
  assert_array_equal(r_a, r_b)

  # n_triplets and n_iter cannot be 0
  msg = ("n_triplets cannot be 0")
  with pytest.raises(ValueError) as raised_error:
    _get_random_indices(n_triplets=0, n_iter=n_iter, random=True)
  assert msg == raised_error.value.args[0]

  msg = ("n_iter cannot be 0")
  with pytest.raises(ValueError) as raised_error:
    _get_random_indices(n_triplets=n_triplets, n_iter=0, random=True)
  assert msg == raised_error.value.args[0]
