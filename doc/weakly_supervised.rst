.. _weakly_supervised_section:

=================================
Weakly Supervised Metric Learning
=================================

Weakly supervised algorithms work on weaker information about the data points
than supervised algorithms. Rather than labeled points, they take as input
similarity judgments on tuples of data points, for instance pairs of similar
and dissimilar points. Refer to the documentation of each algorithm for its
particular form of input data.


General API
===========

Input data
----------

In the following paragraph we talk about tuples for sake of generality. These
can be pairs, triplets, quadruplets etc, depending on the particular metric
learning algorithm we use.

Basic form
^^^^^^^^^^

Every weakly supervised algorithm will take as input tuples of
points, and if needed labels for theses tuples. The tuples of points can
also be called "constraints". They are a set of points that we consider (ex:
two points, three points, etc...). The label is some information we have
about this set of points (e.g. "these two points are similar"). Note that
some information can be contained in the ordering of these tuples (see for
instance the section :ref:`learning_on_quadruplets`). For more details about
specific forms of tuples, refer to the appropriate sections 
(:ref:`learning_on_pairs` or :ref:`learning_on_quadruplets`).

The `tuples` argument is the first argument of every method (like the `X`
argument for classical algorithms in scikit-learn). The second argument is the
label of the tuple: its semantic depends on the algorithm used. For instance
for pairs learners `y` is a label indicating whether the pair is of similar
samples or dissimilar samples.

Then one can fit a Weakly Supervised Metric Learner on this tuple, like this:

>>> my_algo.fit(tuples, y)

Like in a classical setting we split the points `X` between train and test,
here we split the `tuples` between train and test.

>>> from sklearn.model_selection import train_test_split
>>> pairs_train, pairs_test, y_train, y_test = train_test_split(pairs, y)

These are two data structures that can be used to represent tuple in metric
learn:

3D array of tuples
^^^^^^^^^^^^^^^^^^

The most intuitive way to represent tuples is to provide the algorithm with a
3D array-like of tuples of shape `(n_tuples, tuple_size, n_features)`, where
`n_tuples` is the number of tuples, `tuple_size` is the number of elements
in a tuple (2 for pairs, 3 for triplets for instance), and `n_features` is
the number of features of each point.

.. topic:: Example:
   Here is an artificial dataset of 4 pairs of 2 points of 3 features each:

>>> import numpy as np
>>> tuples = np.array([[[-0.12, -1.21, -0.20],
>>>                     [+0.05, -0.19, -0.05]],
>>>
>>>                    [[-2.16, +0.11, -0.02],
>>>                     [+1.58, +0.16, +0.93]],
>>>
>>>                    [[+1.58, +0.16, +0.93],  # same as tuples[1, 1, :]
>>>                     [+0.89, -0.34, +2.41]],
>>>
>>>                    [[-0.12, -1.21, -0.20],  # same as tuples[0, 0, :]
>>>                     [-2.16, +0.11, -0.02]]])  # same as tuples[1, 0, :]
>>> y = np.array([-1, 1, 1, -1])

.. warning:: This way of specifying pairs is not recommended for a large number
   of tuples, as it is redundant (see the comments in the example) and hence
   takes a lot of memory. Indeed each feature vector of a point will be
   replicated as many times as a point is involved in a tuple. The second way
   to specify pairs is more efficient


2D array of indicators + preprocessor
^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^

Instead of forming each point in each tuple, a more efficient representation
would be to keep the dataset of points `X` aside, and just represent tuples
as a collection of tuples of *indices* from the points in `X`. Since we loose
the feature dimension there, the resulting array is 2D.

.. topic:: Example: An equivalent representation of the above pairs would be:

>>> X = np.array([[-0.12, -1.21, -0.20],
>>>               [+0.05, -0.19, -0.05],
>>>               [-2.16, +0.11, -0.02],
>>>               [+1.58, +0.16, +0.93],
>>>               [+0.89, -0.34, +2.41]])
>>>
>>> tuples_indices = np.array([[0, 1],
>>>                            [2, 3],
>>>                            [3, 4],
>>>                            [0, 2]])
>>> y = np.array([-1, 1, 1, -1])

In order to fit metric learning algorithms with this type of input, we need to
give the original dataset of points `X` to the estimator so that it knows
the points the indices refer to. We do this when initializing the estimator,
through the argument `preprocessor` (see below :ref:`fit_ws`)


.. note::

   Instead of an array-like, you can give a callable in the argument
   `preprocessor`, which will go fetch and form the tuples. This allows to
   give more general indicators than just indices from an array (for instance
   paths in the filesystem, name of records in a database etc...) See section
   :ref:`preprocessor_section` for more details on how to use the preprocessor.

.. _fit_ws:

Fit, transform, and so on
-------------------------

The goal of weakly-supervised metric-learning algorithms is to transform
points in a new space, in which the tuple-wise constraints between points
are respected.

>>> from metric_learn import MMC
>>> mmc = MMC(random_state=42)
>>> mmc.fit(tuples, y)
MMC(A0='deprecated', convergence_threshold=0.001, diagonal=False,
  diagonal_c=1.0, init=None, max_iter=100, max_proj=10000,
  preprocessor=None, random_state=42, verbose=False)

Or alternatively (using a preprocessor):

>>> from metric_learn import MMC
>>> mmc = MMC(preprocessor=X, random_state=42)
>>> mmc.fit(pairs_indice, y)


Now that the estimator is fitted, you can use it on new data for several
purposes.

First, you can transform the data in the learned space, using `transform`:
Here we transform two points in the new embedding space.

>>> X_new = np.array([[9.4, 4.1, 4.2], [2.1, 4.4, 2.3]])
>>> mmc.transform(X_new)
array([[-3.24667162e+01,  4.62622348e-07,  3.88325421e-08],
       [-3.61531114e+01,  4.86778289e-07,  2.12654397e-08]])

Also, as explained before, our metric learner has learned a distance between
points. You can use this distance in two main ways:

- You can either return the distance between pairs of points using the
  `score_pairs` function:

>>> mmc.score_pairs([[[3.5, 3.6, 5.2], [5.6, 2.4, 6.7]],
...                  [[1.2, 4.2, 7.7], [2.1, 6.4, 0.9]]])
array([7.27607365, 0.88853014])

- Or you can return a function that will return the distance
  (in the new space) between two 1D arrays (the coordinates of the points in
  the original space), similarly to distance functions in
  `scipy.spatial.distance`. To do that, use the `get_metric` method.

>>> metric_fun = mmc.get_metric()
>>> metric_fun([3.5, 3.6, 5.2], [5.6, 2.4, 6.7])
7.276073646278203

.. note::

    If the metric learner that you use learns a :ref:`Mahalanobis distance
    <mahalanobis_distances>` (like it is the case for all algorithms
    currently in metric-learn), you can get the plain Mahalanobis matrix using
    `get_mahalanobis_matrix`.

>>> mmc.get_mahalanobis_matrix()
array([[ 0.58603894, -5.69883982, -1.66614919],
       [-5.69883982, 55.41743549, 16.20219519],
       [-1.66614919, 16.20219519,  4.73697721]])

.. TODO: remove the "like it is the case etc..." if it's not the case anymore

.. _sklearn_compat_ws:

Prediction and scoring
----------------------

Since weakly supervised are also able, after being fitted, to predict for a
given tuple what is its label (for pairs) or ordering (for quadruplets). See
the appropriate section for more details, either :ref:`this
one <pairs_predicting>` for pairs, or :ref:`this one
<quadruplets_predicting>` for quadruplets.

They also implement a default scoring method, `score`, that can be
used to evaluate the performance of a metric-learner on a test dataset. See
the appropriate section for more details, either :ref:`this
one <pairs_scoring>` for pairs, or :ref:`this one <learning_on_quadruplets>`
for quadruplets.

Scikit-learn compatibility
--------------------------

Weakly supervised estimators are compatible with scikit-learn routines for
model selection (`sklearn.model_selection.cross_val_score`,
`sklearn.model_selection.GridSearchCV`, etc).

Example:

>>> from metric_learn import MMC
>>> import numpy as np
>>> from sklearn.datasets import load_iris
>>> from sklearn.model_selection import cross_val_score
>>> rng = np.random.RandomState(42)
>>> X, _ = load_iris(return_X_y=True)
>>> # let's sample 30 random pairs and labels of pairs
>>> pairs_indices = rng.randint(X.shape[0], size=(30, 2))
>>> y = 2 * rng.randint(2, size=30) - 1
>>> mmc = MMC(preprocessor=X)
>>> cross_val_score(mmc, pairs_indices, y)

.. _learning_on_pairs:

Learning on pairs
=================

Some metric learning algorithms learn on pairs of samples. In this case, one
should provide the algorithm with `n_samples` pairs of points, with a
corresponding target containing `n_samples` values being either +1 or -1.
These values indicate whether the given pairs are similar points or
dissimilar points.

Fitting
-------
Here is an example for fitting on pairs (see :ref:`fit_ws` for more details on
the input data format and how to fit, in the general case of learning on
tuples).

>>> from metric_learn import MMC
>>> pairs = np.array([[[1.2, 3.2], [2.3, 5.5]],
>>>                   [[4.5, 2.3], [2.1, 2.3]]])
>>> y_pairs = np.array([1, -1])
>>> mmc = MMC(random_state=42)
>>> mmc.fit(pairs, y_pairs)
MMC(A0='deprecated', convergence_threshold=0.001, diagonal=False,
    diagonal_c=1.0, init=None, max_iter=100, max_proj=10000, preprocessor=None,
    random_state=42, verbose=False)

Here, we learned a metric that puts the two first points closer
together in the transformed space, and the two next points further away from
each other.

.. _pairs_predicting:

Prediction
----------

When a pairs learner is fitted, it is also able to predict, for an unseen
pair, whether it is a pair of similar or dissimilar points.

>>> mmc.predict([[[0.6, 1.6], [1.15, 2.75]],
...              [[3.2, 1.1], [5.4, 6.1]]])
array([1, -1])

.. _calibration:

Prediction threshold
^^^^^^^^^^^^^^^^^^^^

Predicting whether a new pair represents similar or dissimilar
samples requires to set a threshold on the learned distance, so that points
closer (in the learned space) than this threshold are predicted as similar,
and points further away are predicted as dissimilar. Several methods are
possible for this thresholding.

- **Calibration at fit time**: The threshold is set with `calibrate_threshold`
  (see below) on the training set. You can specify the calibration
  parameters directly
  in the `fit` method with the `threshold_params` parameter (see the
  documentation of the `fit` method of any metric learner that learns on pairs
  of points for more information). Note that calibrating on the training set
  may cause some overfitting. If you want to avoid that, calibrate the
  threshold after fitting, on a validation set.

  >>> mmc.fit(pairs, y) # will fit the threshold automatically after fitting

- **Calibration on validation set**: calling `calibrate_threshold` will
  calibrate the threshold to achieve a particular score on a validation set,
  the score being among the classical scores for classification (accuracy, f1
  score...).

  >>> mmc.calibrate_threshold(pairs, y)

- **Manual threshold**: calling `set_threshold` will set the threshold to a
  particular value.

  >>> mmc.set_threshold(0.4)

See also: `sklearn.calibration`.

.. _pairs_scoring:

Scoring
-------

Pair metric learners can also return a `decision_function` for a set of pairs.
It is basically the "score" that will be thresholded to find the prediction
for the pair. This score corresponds to the opposite of the distance in the
new space (higher score means points are similar, and lower score dissimilar).

>>> mmc.decision_function([[[0.6, 1.6], [1.15, 2.75]],
...                        [[3.2, 1.1], [5.4, 6.1]]])
array([-0.12811124, -0.74750256])

This allows to use common scoring functions for binary classification, like
`sklearn.metrics.accuracy_score` for instance, which
can be used inside cross-validation routines:

>>> from sklearn.model_selection import cross_val_score
>>> pairs_test = np.array([[[0.6, 1.6], [1.15, 2.75]],
...                        [[3.2, 1.1], [5.4, 6.1]],
...                        [[7.7, 5.6], [1.23, 8.4]]])
>>> y_test = np.array([-1., 1., -1.])
>>> cross_val_score(mmc, pairs_test, y_test, scoring='accuracy')
array([1., 0., 1.])

Pairs learners also have a default score, which basically
returns the `sklearn.metrics.roc_auc_score` (which is threshold-independent).

>>> pairs_test = np.array([[[0.6, 1.6], [1.15, 2.75]],
...                        [[3.2, 1.1], [5.4, 6.1]],
...                        [[7.7, 5.6], [1.23, 8.4]]])
>>> y_test = np.array([1., -1., -1.])
>>> mmc.score(pairs_test, y_test)
1.0

.. note::
   See :ref:`fit_ws` for more details on metric learners functions that are
   not specific to learning on pairs, like `transform`, `score_pairs`,
   `get_metric` and `get_mahalanobis_matrix`.

Algorithms
----------

.. _itml:

:py:class:`ITML <metric_learn.ITML>`
^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^

Information Theoretic Metric Learning (:py:class:`ITML <metric_learn.ITML>`)

`ITML` minimizes the (differential) relative entropy, aka Kullback–Leibler 
divergence, between two multivariate Gaussians subject to constraints on the 
associated Mahalanobis distance, which can be formulated into a Bregman 
optimization problem by minimizing the LogDet divergence subject to 
linear constraints. This algorithm can handle a wide variety of constraints
and can optionally incorporate a prior on the distance function. Unlike some
other methods, `ITML` does not rely on an eigenvalue computation or 
semi-definite programming.


Given a Mahalanobis distance parameterized by :math:`M`, its corresponding 
multivariate Gaussian is denoted as:

.. math::
    p(\mathbf{x}; \mathbf{M}) = \frac{1}{Z}\exp(-\frac{1}{2}d_\mathbf{M}
    (\mathbf{x}, \mu)) 
    =  \frac{1}{Z}\exp(-\frac{1}{2}((\mathbf{x} - \mu)^T\mathbf{M}
    (\mathbf{x} - \mu)) 

where :math:`Z` is the normalization constant, the inverse of Mahalanobis 
matrix :math:`\mathbf{M}^{-1}` is the covariance of the Gaussian.

Given pairs of similar points :math:`S` and pairs of dissimilar points 
:math:`D`, the distance metric learning problem is to minimize the LogDet
divergence, which is equivalent as minimizing :math:`\textbf{KL}(p(\mathbf{x}; 
\mathbf{M}_0) || p(\mathbf{x}; \mathbf{M}))`:

.. math::

    \min_\mathbf{A} D_{\ell \mathrm{d}}\left(M, M_{0}\right) = 
    \operatorname{tr}\left(M M_{0}^{-1}\right)-\log \operatorname{det}
    \left(M M_{0}^{-1}\right)-n\\
    \text{subject to } \quad d_\mathbf{M}(\mathbf{x}_i, \mathbf{x}_j) 
    \leq u \qquad (\mathbf{x}_i, \mathbf{x}_j)\in S \\
    d_\mathbf{M}(\mathbf{x}_i, \mathbf{x}_j) \geq l \qquad (\mathbf{x}_i, 
    \mathbf{x}_j)\in D


where :math:`u` and :math:`l` is the upper and the lower bound of distance
for similar and dissimilar pairs respectively, and :math:`\mathbf{M}_0` 
is the prior distance metric, set to identity matrix by default, 
:math:`D_{\ell \mathrm{d}}(\cdot)` is the log determinant.

.. topic:: Example Code:

::

    from metric_learn import ITML

    pairs = [[[1.2, 7.5], [1.3, 1.5]],
             [[6.4, 2.6], [6.2, 9.7]],
             [[1.3, 4.5], [3.2, 4.6]],
             [[6.2, 5.5], [5.4, 5.4]]]
    y = [1, 1, -1, -1]

    # in this task we want points where the first feature is close to be closer
    # to each other, no matter how close the second feature is


    itml = ITML()
    itml.fit(pairs, y)

.. topic:: References:

    .. [1] Jason V. Davis, et al. `Information-theoretic Metric Learning <https://icml.cc/imls/conferences/2007/proceedings/papers/404.pdf>`_. ICML 2007

    .. [2] Adapted from Matlab code at http://www.cs.utexas.edu/users/pjain/itml/


.. _sdml:

:py:class:`SDML <metric_learn.SDML>`
^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^

Sparse High-Dimensional Metric Learning
(:py:class:`SDML <metric_learn.SDML>`)

`SDML` is an efficient sparse metric learning in high-dimensional space via 
double regularization: an L1-penalization on the off-diagonal elements of the 
Mahalanobis matrix :math:`\mathbf{M}`, and a log-determinant divergence between 
:math:`\mathbf{M}` and :math:`\mathbf{M_0}` (set as either :math:`\mathbf{I}` 
or :math:`\mathbf{\Omega}^{-1}`, where :math:`\mathbf{\Omega}` is the 
covariance matrix).

The formulated optimization on the semidefinite matrix :math:`\mathbf{M}` 
is convex:

.. math::

    \min_{\mathbf{M}} = \text{tr}((\mathbf{M}_0 + \eta \mathbf{XLX}^{T})
    \cdot \mathbf{M}) - \log\det \mathbf{M} + \lambda ||\mathbf{M}||_{1, off}

where :math:`\mathbf{X}=[\mathbf{x}_1, \mathbf{x}_2, ..., \mathbf{x}_n]` is 
the training data, the incidence matrix :math:`\mathbf{K}_{ij} = 1` if 
:math:`(\mathbf{x}_i, \mathbf{x}_j)` is a similar pair, otherwise -1. The 
Laplacian matrix :math:`\mathbf{L}=\mathbf{D}-\mathbf{K}` is calculated from 
:math:`\mathbf{K}` and :math:`\mathbf{D}`, a diagonal matrix whose entries are 
the sums of the row elements of :math:`\mathbf{K}`., :math:`||\cdot||_{1, off}` 
is the off-diagonal L1 norm.


.. topic:: Example Code:

::

    from metric_learn import SDML

    pairs = [[[1.2, 7.5], [1.3, 1.5]],
             [[6.4, 2.6], [6.2, 9.7]],
             [[1.3, 4.5], [3.2, 4.6]],
             [[6.2, 5.5], [5.4, 5.4]]]
    y = [1, 1, -1, -1]

    # in this task we want points where the first feature is close to be closer
    # to each other, no matter how close the second feature is

    sdml = SDML()
    sdml.fit(pairs, y)

.. topic:: References:

    .. [1] Qi et al.
       `An efficient sparse metric learning in high-dimensional space via
       L1-penalized log-determinant regularization <https://icml.cc/Conferences/2009/papers/46.pdf>`_.
       ICML 2009.

    .. [2] Adapted from https://gist.github.com/kcarnold/5439945

.. _rca:

:py:class:`RCA <metric_learn.RCA>`
^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^

Relative Components Analysis (:py:class:`RCA <metric_learn.RCA>`)

`RCA` learns a full rank Mahalanobis distance metric based on a weighted sum of
in-chunklets covariance matrices. It applies a global linear transformation to 
assign large weights to relevant dimensions and low weights to irrelevant 
dimensions. Those relevant dimensions are estimated using "chunklets", subsets 
of points that are known to belong to the same class.

For a training set with :math:`n` training points in :math:`k` chunklets, the 
algorithm is efficient since it simply amounts to computing

.. math::

      \mathbf{C} = \frac{1}{n}\sum_{j=1}^k\sum_{i=1}^{n_j}
      (\mathbf{x}_{ji}-\hat{\mathbf{m}}_j)
      (\mathbf{x}_{ji}-\hat{\mathbf{m}}_j)^T


where chunklet :math:`j` consists of :math:`\{\mathbf{x}_{ji}\}_{i=1}^{n_j}` 
with a mean :math:`\hat{m}_j`. The inverse of :math:`\mathbf{C}^{-1}` is used 
as the Mahalanobis matrix.

.. topic:: Example Code:

::

    from metric_learn import RCA

    X = [[-0.05,  3.0],[0.05, -3.0],
        [0.1, -3.55],[-0.1, 3.55],
        [-0.95, -0.05],[0.95, 0.05],
        [0.4,  0.05],[-0.4, -0.05]]
    chunks = [0, 0, 1, 1, 2, 2, 3, 3]

    rca = RCA()
    rca.fit(X, chunks)

.. topic:: References:

    .. [1] Shental et al. `Adjustment learning and relevant component analysis
       <http://citeseerx.ist.psu.edu/viewdoc/download?doi=10.1.1.19.2871
       &rep=rep1&type=pdf>`_. ECCV 2002

    .. [2] Bar-Hillel et al. `Learning distance functions using equivalence relations <https://aaai.org/Papers/ICML/2003/ICML03-005.pdf>`_. ICML 2003

    .. [3] Bar-Hillel et al. `Learning a Mahalanobis metric from equivalence constraints <http://www.jmlr.org/papers/volume6/bar-hillel05a/bar-hillel05a.pdf>`_. JMLR 2005

.. _mmc:

:py:class:`MMC <metric_learn.MMC>`
^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^

Metric Learning with Application for Clustering with Side Information
(:py:class:`MMC <metric_learn.MMC>`)

`MMC` minimizes the sum of squared distances between similar points, while
enforcing the sum of distances between dissimilar ones to be greater than one. 
This leads to a convex and, thus, local-minima-free optimization problem that 
can be solved efficiently. 
However, the algorithm involves the computation of eigenvalues, which is the 
main speed-bottleneck. Since it has initially been designed for clustering 
applications, one of the implicit assumptions of MMC is that all classes form 
a compact set, i.e., follow a unimodal distribution, which restricts the 
possible use-cases of this method. However, it is one of the earliest and a 
still often cited technique.

The algorithm aims at minimizing the sum of distances between all the similar 
points, while constrains the sum of distances between dissimilar points:

.. math::

      \min_{\mathbf{M}\in\mathbb{S}_+^d}\sum_{(\mathbf{x}_i, 
      \mathbf{x}_j)\in S} d_{\mathbf{M}}(\mathbf{x}_i, \mathbf{x}_j)
      \qquad \qquad \text{s.t.} \qquad \sum_{(\mathbf{x}_i, \mathbf{x}_j)
      \in D} d^2_{\mathbf{M}}(\mathbf{x}_i, \mathbf{x}_j) \geq 1

.. topic:: Example Code:

::

    from metric_learn import MMC

    pairs = [[[1.2, 7.5], [1.3, 1.5]],
             [[6.4, 2.6], [6.2, 9.7]],
             [[1.3, 4.5], [3.2, 4.6]],
             [[6.2, 5.5], [5.4, 5.4]]]
    y = [1, 1, -1, -1]

    # in this task we want points where the first feature is close to be closer
    # to each other, no matter how close the second feature is

    mmc = MMC()
    mmc.fit(pairs, y)

.. topic:: References:

  .. [1] Xing et al. `Distance metric learning with application to clustering with
        side-information <http://papers.nips
        .cc/paper/2164-distance-metric-learning-with-application-to-clustering
        -with-side-information.pdf>`_. NIPS 2002
  .. [2] Adapted from Matlab code http://www.cs.cmu.edu/%7Eepxing/papers/Old_papers/code_Metric_online.tar.gz


.. _learning_on_quadruplets:

Learning on quadruplets
=======================

Some metric learning algorithms learn on quadruplets of samples. In this case,
one should provide the algorithm with `n_samples` quadruplets of points. Th
semantic of each quadruplet is that the first two points should be closer
together than the last two points.

Fitting
-------
Here is an example for fitting on quadruplets (see :ref:`fit_ws` for more
details on the input data format and how to fit, in the general case of
learning on tuples).

>>> from metric_learn import LSML
>>> quadruplets = np.array([[[1.2, 3.2], [2.3, 5.5], [2.4, 6.7], [2.1, 0.6]],
>>>                         [[4.5, 2.3], [2.1, 2.3], [0.6, 1.2], [7.3, 3.4]]])
>>> lsml = LSML(random_state=42)
>>> lsml.fit(quadruplets)
LSML(max_iter=1000, preprocessor=None, prior=None, random_state=42, tol=0.001,
   verbose=False)

Or alternatively (using a preprocessor):

>>> X = np.array([[1.2, 3.2],
>>>               [2.3, 5.5],
>>>               [2.4, 6.7],
>>>               [2.1, 0.6],
>>>               [4.5, 2.3],
>>>               [2.1, 2.3],
>>>               [0.6, 1.2],
>>>               [7.3, 3.4]])
>>> quadruplets_indices = np.array([[0, 1, 2, 3], [4, 5, 6, 7]])
>>> lsml = LSML(preprocessor=X, random_state=42)
>>> lsml.fit(quadruplets_indices)
LSML(max_iter=1000,
   preprocessor=array([[1.2, 3.2],
       [2.3, 5.5],
       [2.4, 6.7],
       [2.1, 0.6],
       [4.5, 2.3],
       [2.1, 2.3],
       [0.6, 1.2],
       [7.3, 3.4]]),
   prior=None, random_state=42, tol=0.001, verbose=False)


Here, we want to learn a metric that, for each of the two
`quadruplets`, will put the two first points closer together than the two
last points.

.. _quadruplets_predicting:

Prediction
----------

When a quadruplets learner is fitted, it is also able to predict, for an
upcoming quadruplet, whether the two first points are more similar than the
two last points (+1), or not (-1).

>>> quadruplets_test = np.array(
... [[[5.6, 5.3], [2.2, 2.1], [0.4, 0.6], [1.2, 3.4]],
...  [[6.0, 4.2], [4.3, 1.2], [4.5, 0.6], [0.1, 7.8]]])
>>> lsml.predict(quadruplets_test)
array([-1.,  1.])

.. _quadruplets_scoring:

Scoring
-------

Quadruplet metric learners can also
return a `decision_function` for a set of pairs. This is basically the "score"
which sign will be taken to find the prediction for the pair, which
corresponds to the difference between the distance between the two last points,
and the distance between the two last points of the quadruplet (higher
score means the two last points are more likely to be more dissimilar than
the two first points (i.e. more likely to have a +1 prediction since it's
the right ordering)).

>>> lsml.decision_function(quadruplets_test)
array([-1.75700306,  4.98982131])

In the above example, for the first quadruplet in `quadruplets_test`, the
two first points are predicted less similar than the two last points (they
are further away in the transformed space).

Unlike for pairs learners, quadruplets learners don't allow to give a `y`
when fitting, which does not allow to use scikit-learn scoring functions
like:

>>> from sklearn.model_selection import cross_val_score
>>> cross_val_score(lsml, quadruplets, scoring='f1_score')  # this won't work

(This is actually intentional, for more details
about that, see
`this comment <https://github.com/scikit-learn-contrib/metric-learn/pull/168#pullrequestreview-203730742>`_
on github.)

However, quadruplets learners do have a default scoring function, which will
basically return the accuracy score on a given test set, i.e. the proportion
of quadruplets have the right predicted ordering.

>>> lsml.score(quadruplets_test)
0.5

.. note::
   See :ref:`fit_ws` for more details on metric learners functions that are
   not specific to learning on pairs, like `transform`, `score_pairs`,
   `get_metric` and `get_mahalanobis_matrix`.




Algorithms
----------

.. _lsml:

:py:class:`LSML <metric_learn.LSML>`
^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^

Metric Learning from Relative Comparisons by Minimizing Squared Residual
(:py:class:`LSML <metric_learn.LSML>`)

`LSML` proposes a simple, yet effective, algorithm that minimizes a convex 
objective function corresponding to the sum of squared residuals of 
constraints. This algorithm uses the constraints in the form of the 
relative distance comparisons, such method is especially useful where 
pairwise constraints are not natural to obtain, thus pairwise constraints 
based algorithms become infeasible to be deployed. Furthermore, its sparsity 
extension leads to more stable estimation when the dimension is high and 
only a small amount of constraints is given.

The loss function of each constraint 
:math:`d(\mathbf{x}_a, \mathbf{x}_b) < d(\mathbf{x}_c, \mathbf{x}_d)` is 
denoted as:

.. math::

    H(d_\mathbf{M}(\mathbf{x}_a, \mathbf{x}_b) 
    - d_\mathbf{M}(\mathbf{x}_c, \mathbf{x}_d))

where :math:`H(\cdot)` is the squared Hinge loss function defined as:

.. math::

    H(x) = \left\{\begin{aligned}0 \qquad x\leq 0 \\
    \,\,x^2 \qquad x>0\end{aligned}\right.\\

The summed loss function :math:`L(C)` is the simple sum over all constraints 
:math:`C = \{(\mathbf{x}_a , \mathbf{x}_b , \mathbf{x}_c , \mathbf{x}_d) 
: d(\mathbf{x}_a , \mathbf{x}_b) < d(\mathbf{x}_c , \mathbf{x}_d)\}`. The 
original paper suggested here should be a weighted sum since the confidence 
or probability of each constraint might differ. However, for the sake of 
simplicity and assumption of no extra knowledge provided, we just deploy 
the simple sum here as well as what the authors did in the experiments.

The distance metric learning problem becomes minimizing the summed loss 
function of all constraints plus a regularization term w.r.t. the prior 
knowledge:

.. math::

    \min_\mathbf{M}(D_{ld}(\mathbf{M, M_0}) + \sum_{(\mathbf{x}_a, 
    \mathbf{x}_b, \mathbf{x}_c, \mathbf{x}_d)\in C}H(d_\mathbf{M}(
    \mathbf{x}_a, \mathbf{x}_b) - d_\mathbf{M}(\mathbf{x}_c, \mathbf{x}_c))\\

where :math:`\mathbf{M}_0` is the prior metric matrix, set as identity 
by default, :math:`D_{ld}(\mathbf{\cdot, \cdot})` is the LogDet divergence:

.. math::

    D_{ld}(\mathbf{M, M_0}) = \text{tr}(\mathbf{MM_0}) − \text{logdet}
    (\mathbf{M})

.. topic:: Example Code:

::

    from metric_learn import LSML

    quadruplets = [[[1.2, 7.5], [1.3, 1.5], [6.4, 2.6], [6.2, 9.7]],
                   [[1.3, 4.5], [3.2, 4.6], [6.2, 5.5], [5.4, 5.4]],
                   [[3.2, 7.5], [3.3, 1.5], [8.4, 2.6], [8.2, 9.7]],
                   [[3.3, 4.5], [5.2, 4.6], [8.2, 5.5], [7.4, 5.4]]]

    # we want to make closer points where the first feature is close, and
    # further if the second feature is close

    lsml = LSML()
    lsml.fit(quadruplets)

.. topic:: References:

    .. [1] Liu et al.
       `Metric Learning from Relative Comparisons by Minimizing Squared
       Residual <http://www.cs.ucla.edu/~weiwang/paper/ICDM12.pdf>`_. ICDM 2012

    .. [2] Adapted from https://gist.github.com/kcarnold/5439917


