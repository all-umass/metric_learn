import numpy as np
import six
from sklearn.utils import check_array
from sklearn.utils.validation import check_X_y
from metric_learn.exceptions import PreprocessorError

# hack around lack of axis kwarg in older numpy versions
try:
  np.linalg.norm([[4]], axis=1)
except TypeError:
  def vector_norm(X):
    return np.apply_along_axis(np.linalg.norm, 1, X)
else:
  def vector_norm(X):
    return np.linalg.norm(X, axis=1)


def check_input(input_data, y=None, preprocessor=None,
                type_of_inputs='classic', tuple_size=None, accept_sparse=False,
                dtype='numeric', order=None,
                copy=False, force_all_finite=True,
                multi_output=False, ensure_min_samples=1,
                ensure_min_features=1, y_numeric=False,
                warn_on_dtype=False, estimator=None):
  """Checks that the input format is valid, and converts it if specified
  (this is the equivalent of scikit-learn's `check_array` or `check_X_y`).
  All arguments following tuple_size are scikit-learn's `check_X_y`
  arguments that will be enforced on the data and labels array. If
  indicators are given as an input data array, the returned data array
  will be the formed points/tuples, using the given preprocessor.

  Parameters
  ----------
  input : array-like
    The input data array to check.

  y : array-like
    The input labels array to check.

  preprocessor : callable (default=`None`)
    The preprocessor to use. If None, no preprocessor is used.

  type_of_inputs : `str` {'classic', 'tuples'}
    The type of inputs to check. If 'classic', the input should be
    a 2D array-like of points or a 1D array like of indicators of points. If
    'tuples', the input should be a 3D array-like of tuples or a 2D
    array-like of indicators of tuples.

  accept_sparse : `bool`
    Set to true to allow sparse inputs (only works for sparse inputs with
    dim < 3).

  tuple_size : int
    The number of elements in a tuple (e.g. 2 for pairs).

  dtype : string, type, list of types or None (default='numeric')
    Data type of result. If None, the dtype of the input is preserved.
    If 'numeric', dtype is preserved unless array.dtype is object.
    If dtype is a list of types, conversion on the first type is only
    performed if the dtype of the input is not in the list.

  order : 'F', 'C' or None (default=`None`)
    Whether an array will be forced to be fortran or c-style.

  copy : boolean (default=False)
    Whether a forced copy will be triggered. If copy=False, a copy might
    be triggered by a conversion.

  force_all_finite : boolean or 'allow-nan', (default=True)
    Whether to raise an error on np.inf and np.nan in X. This parameter
    does not influence whether y can have np.inf or np.nan values.
    The possibilities are:
     - True: Force all values of X to be finite.
     - False: accept both np.inf and np.nan in X.
     - 'allow-nan':  accept  only  np.nan  values in  X.  Values  cannot  be
       infinite.

  ensure_min_samples : int (default=1)
    Make sure that X has a minimum number of samples in its first
    axis (rows for a 2D array).

  ensure_min_features : int (default=1)
    Make sure that the 2D array has some minimum number of features
    (columns). The default value of 1 rejects empty datasets.
    This check is only enforced when X has effectively 2 dimensions or
    is originally 1D and ``ensure_2d`` is True. Setting to 0 disables
    this check.

  warn_on_dtype : boolean (default=False)
    Raise DataConversionWarning if the dtype of the input data structure
    does not match the requested dtype, causing a memory copy.

  estimator : str or estimator instance (default=`None`)
    If passed, include the name of the estimator in warning messages.

  Returns
  -------
  X : `numpy.ndarray`
    The checked input data array.

  y: `numpy.ndarray` (optional)
    The checked input labels array.
  """

  context = make_context(estimator)

  args_for_sk_checks = dict(accept_sparse=accept_sparse,
                            dtype=dtype, order=order,
                            copy=copy, force_all_finite=force_all_finite,
                            ensure_min_samples=ensure_min_samples,
                            ensure_min_features=ensure_min_features,
                            warn_on_dtype=warn_on_dtype, estimator=estimator)

  # We need to convert input_data into a numpy.ndarray if possible, before
  # any further checks or conversions, and deal with y if needed. Therefore
  # we use check_array/check_X_y with fixed permissive arguments.
  if y is None:
    input_data = check_array(input_data, ensure_2d=False, allow_nd=True,
                             copy=False, force_all_finite=False,
                             accept_sparse=True, dtype=None,
                             ensure_min_features=0, ensure_min_samples=0)
  else:
    input_data, y = check_X_y(input_data, y, ensure_2d=False, allow_nd=True,
                              copy=False, force_all_finite=False,
                              accept_sparse=True, dtype=None,
                              ensure_min_features=0, ensure_min_samples=0,
                              multi_output=multi_output,
                              y_numeric=y_numeric)

  if type_of_inputs == 'classic':
    input_data = check_input_classic(input_data, context, preprocessor,
                                     args_for_sk_checks)

  elif type_of_inputs == 'tuples':
    input_data = check_input_tuples(input_data, context, preprocessor,
                                    args_for_sk_checks, tuple_size)

  else:
    raise ValueError("Unknown value {} for type_of_inputs. Valid values are "
                     "'classic' or 'tuples'.".format(type_of_inputs))

  return input_data if y is None else (input_data, y)


def check_input_tuples(input_data, context, preprocessor, args_for_sk_checks,
                       tuple_size):
  preprocessor_has_been_applied = False
  if input_data.ndim == 2:
    if preprocessor is not None:
      input_data = preprocess_tuples(input_data, preprocessor)
      preprocessor_has_been_applied = True
    else:
      make_error_input(201, input_data, context)
  elif input_data.ndim == 3:
    pass
  else:
    if preprocessor is not None:
      make_error_input(420, input_data, context)
    else:
      make_error_input(200, input_data, context)
  input_data = check_array(input_data, allow_nd=True, ensure_2d=False,
                           **args_for_sk_checks)
  # we need to check num_features because check_array does not check it
  # for 3D inputs:
  if args_for_sk_checks['ensure_min_features'] > 0:
    n_features = input_data.shape[2]
    if n_features < args_for_sk_checks['ensure_min_features']:
      raise ValueError("Found array with {} feature(s) (shape={}) while"
                       " a minimum of {} is required{}."
                       .format(n_features, input_data.shape,
                               args_for_sk_checks['ensure_min_features'],
                               context))
  #  normally we don't need to check_tuple_size too because tuple_size
  # shouldn't be able to be modified by any preprocessor
  if input_data.ndim != 3:
    # we have to ensure this because check_array above does not
    if preprocessor_has_been_applied:
      make_error_input(211, input_data, context)
    else:
      make_error_input(201, input_data, context)
  check_tuple_size(input_data, tuple_size, context)
  return input_data


def check_input_classic(input_data, context, preprocessor, args_for_sk_checks):
  preprocessor_has_been_applied = False
  if input_data.ndim == 1:
    if preprocessor is not None:
      input_data = preprocess_points(input_data, preprocessor)
      preprocessor_has_been_applied = True
    else:
      make_error_input(101, input_data, context)
  elif input_data.ndim == 2:
    pass  # OK
  else:
    if preprocessor is not None:
      make_error_input(320, input_data, context)
    else:
      make_error_input(100, input_data, context)

  input_data = check_array(input_data, allow_nd=True, ensure_2d=False,
                           **args_for_sk_checks)
  if input_data.ndim != 2:
    # we have to ensure this because check_array above does not
    if preprocessor_has_been_applied:
      make_error_input(111, input_data, context)
    else:
      make_error_input(101, input_data, context)
  return input_data


def make_error_input(code, input_data, context):
  code_str = {'expected_input': {'1': '2D array of formed points',
                                 '2': '3D array of formed tuples',
                                 '3': ('1D array of indicators or 2D array of '
                                       'formed points'),
                                 '4': ('2D array of indicators or 3D array '
                                       'of formed tuples')},
              'additional_context': {'0': '',
                                     '2': ' when using a preprocessor',
                                     '1': (' after the preprocessor has been '
                                           'applied')},
              'possible_preprocessor': {'0': '',
                                        '1': ' and/or use a preprocessor'
                                        }}
  code_list = str(code)
  err_args = dict(expected_input=code_str['expected_input'][code_list[0]],
                  additional_context=code_str['additional_context']
                  [code_list[1]],
                  possible_preprocessor=code_str['possible_preprocessor']
                  [code_list[2]],
                  input_data=input_data, context=context,
                  found_size=input_data.ndim)
  err_msg = ('{expected_input} expected'
             '{context}{additional_context}. Found {found_size}D array '
             'instead:\ninput={input_data}. Reshape your data'
             '{possible_preprocessor}.\n')
  raise ValueError(err_msg.format(**err_args))


def preprocess_tuples(tuples, preprocessor):
  try:
    tuples = np.column_stack([preprocessor(tuples[:, i])[:, np.newaxis] for
                              i in range(tuples.shape[1])])
  except Exception as e:
    raise PreprocessorError(e)
  return tuples


def preprocess_points(points, preprocessor):
  """form points if there is a preprocessor else keep them as such (assumes
  that check_points has already been called)"""
  try:
    points = preprocessor(points)
  except Exception as e:
    raise PreprocessorError(e)
  return points


def make_context(estimator):
  """Helper function to create a string with the estimator name.
  Taken from check_array function in scikit-learn.
  Will return the following for instance:
  NCA: ' by NCA'
  'NCA': ' by NCA'
  None: ''
  """
  estimator_name = make_name(estimator)
  context = (' by ' + estimator_name) if estimator_name is not None else ''
  return context


def make_name(estimator):
  """Helper function that returns the name of estimator or the given string
  if a string is given
  """
  if estimator is not None:
    if isinstance(estimator, six.string_types):
      estimator_name = estimator
    else:
      estimator_name = estimator.__class__.__name__
  else:
    estimator_name = None
  return estimator_name


def check_tuple_size(tuples, tuple_size, context):
  """Helper function to check that the number of points in each tuple is
  equal to tuple_size (e.g. 2 for pairs), and raise a `ValueError` otherwise"""
  if tuple_size is not None and tuples.shape[1] != tuple_size:
    msg_t = (("Tuples of {} element(s) expected{}. Got tuples of {} "
             "element(s) instead (shape={}):\ninput={}.\n")
             .format(tuple_size, context, tuples.shape[1], tuples.shape,
                     tuples))
    raise ValueError(msg_t)


class ArrayIndexer:

  def __init__(self, X):
    # we check the array-like preprocessor here, and we as much permissive
    # as possible (because the user will check for the desired
    # format with arguments in check_input, and only this latter function
    # should return the appropriate errors). We do this only to have a numpy
    # array object which can be indexed by another numpy array object.
    X = check_array(X,
                    accept_sparse=True, dtype=None,
                    force_all_finite=False,
                    ensure_2d=False, allow_nd=True,
                    ensure_min_samples=0,
                    ensure_min_features=0,
                    warn_on_dtype=False, estimator=None)
    self.X = X

  def __call__(self, indices):
    return self.X[indices]


def check_collapsed_pairs(pairs):
    num_ident = (vector_norm(pairs[:, 0] - pairs[:, 1]) < 1e-9).sum()
    if num_ident:
      raise ValueError("{} collapsed pairs found (where the left element is "
                       "the same as the right element), out of {} pairs "
                       "in total.".format(num_ident, pairs.shape[0]))
