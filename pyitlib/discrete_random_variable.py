# -*- coding: utf-8 -*-
# The MIT License (MIT)
#
# Copyright (c) 2016 Peter Foster <pyitlib@gmx.us>
#
# Permission is hereby granted, free of charge, to any person obtaining a copy
# of this software and associated documentation files (the "Software"), to deal
# in the Software without restriction, including without limitation the rights
# to use, copy, modify, merge, publish, distribute, sublicense, and/or sell
# copies of the Software, and to permit persons to whom the Software is
# furnished to do so, subject to the following conditions:
#
# The above copyright notice and this permission notice shall be included in
# all copies or substantial portions of the Software.
#
# THE SOFTWARE IS PROVIDED "AS IS", WITHOUT WARRANTY OF ANY KIND, EXPRESS OR
# IMPLIED, INCLUDING BUT NOT LIMITED TO THE WARRANTIES OF MERCHANTABILITY,
# FITNESS FOR A PARTICULAR PURPOSE AND NONINFRINGEMENT. IN NO EVENT SHALL THE
# AUTHORS OR COPYRIGHT HOLDERS BE LIABLE FOR ANY CLAIM, DAMAGES OR OTHER
# LIABILITY, WHETHER IN AN ACTION OF CONTRACT, TORT OR OTHERWISE, ARISING FROM,
# OUT OF OR IN CONNECTION WITH THE SOFTWARE OR THE USE OR OTHER DEALINGS IN THE
# SOFTWARE.

"""
This module implements various information-theoretic quantities for discrete
random variables.

For ease of reference, function names follow the following convention:

Function names beginning with "entropy" : Entropy measures

Function names beginning with "information" : Mutual information measures

Function names beginning with "divergence" : Divergence measures

Function names ending with "pmf" : Functions operating on arrays of probability
mass assignments (as opposed realisations of random variables)

================================================== ===================== ============== ======== ======== =======================
Function                                           Generalises           Non-negativity Symmetry Identity Metric properties
================================================== ===================== ============== ======== ======== =======================
:meth:`divergence_jensenshannon`                                         Yes            Yes      Yes      Square root is a metric
:meth:`divergence_jensenshannon_pmf`                                     Yes            Yes      Yes      Square root is a metric
:meth:`divergence_kullbackleibler`                                       Yes            No       Yes
:meth:`divergence_kullbackleibler_pmf`                                   Yes            No       Yes
:meth:`divergence_kullbackleibler_symmetrised`                           Yes            Yes      Yes
:meth:`divergence_kullbackleibler_symmetrised_pmf`                       Yes            Yes      Yes
:meth:`entropy`                                                          Yes
:meth:`entropy_conditional`                                              Yes            No       No
:meth:`entropy_cross`                                                    Yes            No       No
:meth:`entropy_cross_pmf`                                                Yes            No       No
:meth:`entropy_joint`                                                    Yes            Yes      No
:meth:`entropy_pmf`                                                      Yes
:meth:`entropy_residual`                           information_variation Yes            Yes      No
:meth:`information_binding`                        information_mutual    Yes            Yes      No
:meth:`information_co`                             information_mutual    No             No       No
:meth:`information_enigmatic`                                            No             Yes      No
:meth:`information_exogenous_local`                                      Yes            Yes      No
:meth:`information_interaction`                    information_mutual    No             No       No
:meth:`information_lautum`                                               Yes            No       No
:meth:`information_multi`                          information_mutual    Yes            Yes      No
:meth:`information_mutual`                                               Yes            Yes      No
:meth:`information_mutual_conditional`                                   Yes            No       No
:meth:`information_mutual_normalised`                                    Yes            See docs No       See docs
:meth:`information_variation`                                            Yes            Yes      No       Is a metric
================================================== ===================== ============== ======== ======== =======================

.. rubric:: References
.. [AbPl12] Abdallah, S.A.; Plumbley, M.D.: A measure of statistical \
complexity based on predictive information with application to finite spin \
systems. In: Physics Letters A, Vol. 376, No. 4, 2012, P. 275-281.
.. [Bell03] Bell, A.J.: The co-information lattice. In: Proceedings of the \
International Workshop on Independent Component Analysis and Blind Signal \
Separation. 2003.
.. [CoTh06] Cover, T.M.; Thomas, J.A.: Elements of information theory \
(2nd ed.). John Wiley & Sons, 2006.
.. [Croo15] Crooks, G.E.: On measures of entropy and information. \
http://threeplusone.com/info, retrieved 2017-03-16.
.. [GaSa95] Gale, W.A.; Sampson, G.: Good‐Turing frequency estimation \
without tears. In: Journal of Quantitative Linguistics, \
Vol. 2, No. 3, 1995, P. 217-237.
.. [Han78] Han, T.S.: Nonnegative entropy measures of multivariate symmetric \
correlations. In: Information and Control, Vol. 36, 1978, P. 133-156.
.. [HaSt09] Hausser, J.; Strimmer, K.: Entropy inference and the James-Stein \
estimator, with application to nonlinear gene association networks. \
In: Journal of Machine Learning Research, Vol. 10, 2009, P. 1469-1484.
.. [JaBr03] Jakulin, A.; Bratko, I.: Quantifying and visualizing attribute \
interactions. arXiv preprint cs/0308002, 2003.
.. [JaEC11] James, R.G.; Ellison, C.J.; Crutchfield, J.P.: Anatomy of a bit: \
Information in a time series observation. In: Chaos: An Interdisciplinary \
Journal of Nonlinear Science, Vol. 21, No. 3, 2011.
.. [Lin91] Lin, J.: Divergence measures based on the Shannon entropy. \
In: IEEE Transactions on Information theory, Vol. 37, No. 1, 1991, P. 145-151.
.. [Meil03] Meilă, M.: Comparing clusterings by the variation of \
information. In: Learning theory and kernel machines. \
Springer, 2003, P. 173-187.
.. [Murp12] Murphy, K. P.: Machine learning: a probabilistic perspective. \
MIT press, 2012.
.. [PaVe08] Palomar, D. P.; Verdú, S.: Lautum information. In: IEEE \
transactions on information theory, Vol. 54, No. 3, 2008, P. 964-975.
.. [StVe98] Studený, M.; Vejnarová, J.: The multiinformation function \
as a tool for measuring stochastic dependence. In: Learning in graphical \
models. Springer Netherlands, 1998, P. 261-297.
.. [VeWe06] Verdú, S.; Weissman, T.: Erasure entropy. In: Proc. IEEE \
International Symposium on Information Theory, 2006, P. 98-102.
.. [Wata60] Watanabe, S.: Information theoretical analysis of multivariate \
correlation. In: IBM Journal of research and development, \
Vol. 4, No. 1, 1960, P. 66-82.
"""
from __future__ import division

from builtins import zip
from builtins import str
from builtins import range

import warnings
import numpy as np
import sklearn.preprocessing
import pandas as pd

NONE_REPLACEMENT = -32768

# Aims of project: Comprehensive, Simple-to-use (avoid lots of function calls,
# prefer flags, convenient defaults for possible interactive use). Focus on
# data analysis. Documentation.
# TODO Add guidance on which estimator to use (within module doc)
# TODO Add notes on interpretation of each measure (within each function doc)
# TODO Add basic equalities and properties, followed by interpretation (within
# each function doc)
# TODO Note about which measures are metrics (within each function doc).
# TODO Add information diagrams (within each function doc).
# TODO Implement joint observation mapping function
# encode/map_joint_observations. This works by sorting and returning unique
# observations, similar to entropy_joint()
# TODO Implement generalised Jensen-Shannon divergence
# TODO Information bottleneck and deterministic information bottleneck (See
# Strose and Schwab 2016). NB Can be used in either supervised or unsupervised
# manner. Implement in separate module. Implement the earlier hierarchical
# aggolerative clustering approach as well?. The approaches should also
# incorporate `de-noising' capability: With probability p, corrupt observations
# with a random symbol. The learnt model should also have a way of discarding
# un-used input features (feature selection).
# TODO Re-arrange functions based on dependencies
# TODO Add information in documentation on when quantities are maximised or
# minimised


def entropy_residual(X, base=2, fill_value=-1, estimator='ML',
                     Alphabet_X=None, keep_dims=False):
    """
    Returns the estimated residual entropy [JaEC11] (also known as erasure
    entropy [VeWe06]) for an array X containing realisations of discrete random
    variables.

    **Mathematical definition**:

    Given discrete random variables :math:`X_1, \ldots, X_n`, the residual
    entropy :math:`R(X_1, \\ldots, X_n)` is defined as:

    .. math::
        R(X_1, \\ldots, X_n) = H(X_1, \\ldots, X_n) - B(X_1, \\ldots, X_n)

    where :math:`H(\\cdot, \\ldots, \\cdot)` denotes the joint entropy and
    where :math:`B(\\cdot, \\ldots, \\cdot)` denotes the binding information.

    **Estimation**:

    Residual information is estimated based on frequency tables, using the
    following functions:

        entropy_joint()

        entropy()

    See below for a list of available estimators. Note that although residual
    information is a non-negative quantity, depending on the chosen estimator
    the obtained estimate may be negative.

    **Parameters**:

    X : numpy array (or array-like object such as a list of immutables, as \
    accepted by np.array())
        An array containing discrete random variable realisations. Successive
        realisations of a random variable are indexed by the last axis in the
        array; multiple random variables may be specified using preceding axes.
        When X.ndim==1, returns a scalar and is equivalent to entropy(). When
        X.ndim>1, returns a scalar based on jointly considering all random
        variables indexed in the array. X may not contain (floating point) NaN
        values. Missing data may be specified using numpy masked arrays, as
        well as using standard numpy array/array-like objects; see below
        for details.
    base : float
        The desired logarithmic base (default 2).
    fill_value : object
        It is possible to specify missing data using numpy masked arrays,
        pandas Series/DataFrames, as well as using standard numpy
        array/array-like objects with assigned placeholder values. When using
        numpy masked arrays, this function invokes np.ma.filled() internally,
        so that missing data are represented with the array's object-internal
        placeholder value fill_value (this function's fill_value parameter is
        ignored in such cases). When using pandas Series/DataFrames, an initial
        conversion to a numpy masked array is performed. When using standard
        numpy array/array-like objects, this function's fill_value parameter is
        used to specify the placeholder value for missing data (defaults to
        -1).

        Data equal to the placeholder value are subsequently ignored.
    estimator : str or float
        The desired estimator (see above for details on estimators). Possible
        values are:

            *'ML' (the default value)* : Maximum likelihood estimator.

            *any floating point value* : Maximum a posteriori esimator using
            Dirichlet prior (equivalent to maximum likelihood with pseudo-count
            for each outcome as specified).

            *PERKS* : Maximum a posteriori esimator using Dirichlet prior
            (equivalent to maximum likelihood with pseudo-count for each
            outcome set to 1/L, where L is the number of possible outcomes.

            *MINIMAX* : Maximum a posteriori esimator using Dirichlet prior
            (equivalent to maximum likelihood with pseudo-count for each
            outcome set to sqrt(N)/L, where N is the total number of
            realisations and where L is the number of possible outcomes.

            *JAMES-STEIN* : James-Stein estimator [HaSt09].

            *GOOD-TURING* : Good-Turing estimator [GaSa95].

    Alphabet_X : numpy array (or array-like object such as a list of \
    immutables, as accepted by np.array())
        An array specifying the alphabet/alphabets of possible outcomes that
        random variable realisations in array X may assume. Defaults to None,
        in which case the alphabet/alphabets of possible outcomes is/are
        implicitly based the observed outcomes in array X, with no additional,
        unobserved outcomes. In combination with any estimator other than
        maximum likelihood, it may be useful to specify alphabets including
        unobserved outcomes. For such cases, successive possible outcomes of a
        random variable are indexed by the last axis in Alphabet_X; multiple
        alphabets may be specified using preceding axes, with the requirement
        X.shape[:-1]==Alphabet_X.shape[:-1]. Alphabets of different sizes may
        be specified either using numpy masked arrays, or by padding with the
        chosen placeholder fill_value.

        NB: When specifying multiple alphabets, an alphabet of possible joint
        outcomes is always implicit from the alphabets of possible (marginal)
        outcomes in Alphabet_X. For example, specifying
        Alphabet_X=np.array(((1,2),(1,2))) implies an alphabet of possible
        joint outcomes np.array(((1,1,2,2),(1,2,1,2))).
    keep_dims : boolean
        When set to True, an additional dimension of length one is appended to
        the returned array, facilitating any broadcast operations required by
        the user (defaults to False).

    **Implementation notes**:

    Before estimation, outcomes are mapped to the set of non-negative integers
    internally, with the value -1 representing missing data. To avoid this
    internal conversion step, supply integer data and use the default fill
    value -1.

    """
    H_joint = entropy_joint(X, base, fill_value, estimator, Alphabet_X)

    R = H_joint - information_binding(X, base, fill_value, estimator,
                                      Alphabet_X)

    if keep_dims:
        R = R[..., np.newaxis]

    return R


def information_exogenous_local(X, base=2, fill_value=-1, estimator='ML',
                                Alphabet_X=None, keep_dims=False):
    """
    Returns the estimated exogenous local information [JaEC11] for an array X
    containing realisations of discrete random variables.

    **Mathematical definition**:

    Given discrete random variables :math:`X_1, \ldots, X_n`, the exogenous
    local information :math:`W(X_1, \\ldots, X_n)` is defined as:

    .. math::
        W(X_1, \\ldots, X_n) = T(X_1, \\ldots, X_n) + B(X_1, \\ldots, X_n)

    where :math:`T(\\cdot, \\ldots, \\cdot)` denotes the multi-information and
    where :math:`B(\\cdot, \\ldots, \\cdot)` denotes the binding information.

    **Estimation**:

    Exogenous local information is estimated based on frequency tables, using
    the following functions:

        entropy_joint()

        entropy()

    See below for a list of available estimators. Note that although exogenous
    local information is a non-negative quantity, depending on the chosen
    estimator the obtained estimate may be negative.

    **Parameters**:

    X : numpy array (or array-like object such as a list of immutables, as \
    accepted by np.array())
        An array containing discrete random variable realisations. Successive
        realisations of a random variable are indexed by the last axis in the
        array; multiple random variables may be specified using preceding axes.
        When X.ndim==1, returns the scalar 0. When X.ndim>1, returns a scalar
        based on jointly considering all random variables indexed in the array.
        X may not contain (floating point) NaN values. Missing data may be
        specified using numpy masked arrays, as well as using standard
        numpy array/array-like objects; see below for details.
    base : float
        The desired logarithmic base (default 2).
    fill_value : object
        It is possible to specify missing data using numpy masked arrays,
        pandas Series/DataFrames, as well as using standard numpy
        array/array-like objects with assigned placeholder values. When using
        numpy masked arrays, this function invokes np.ma.filled() internally,
        so that missing data are represented with the array's object-internal
        placeholder value fill_value (this function's fill_value parameter is
        ignored in such cases). When using pandas Series/DataFrames, an initial
        conversion to a numpy masked array is performed. When using standard
        numpy array/array-like objects, this function's fill_value parameter is
        used to specify the placeholder value for missing data (defaults to
        -1).

        Data equal to the placeholder value are subsequently ignored.
    estimator : str or float
        The desired estimator (see above for details on estimators). Possible
        values are:

            *'ML' (the default value)* : Maximum likelihood estimator.

            *any floating point value* : Maximum a posteriori esimator using
            Dirichlet prior (equivalent to maximum likelihood with pseudo-count
            for each outcome as specified).

            *PERKS* : Maximum a posteriori esimator using Dirichlet prior
            (equivalent to maximum likelihood with pseudo-count for each
            outcome set to 1/L, where L is the number of possible outcomes.

            *MINIMAX* : Maximum a posteriori esimator using Dirichlet prior
            (equivalent to maximum likelihood with pseudo-count for each
            outcome set to sqrt(N)/L, where N is the total number of
            realisations and where L is the number of possible outcomes.

            *JAMES-STEIN* : James-Stein estimator [HaSt09].

            *GOOD-TURING* : Good-Turing estimator [GaSa95].

    Alphabet_X : numpy array (or array-like object such as a list of \
    immutables, as accepted by np.array())
        An array specifying the alphabet/alphabets of possible outcomes that
        random variable realisations in array X may assume. Defaults to None,
        in which case the alphabet/alphabets of possible outcomes is/are
        implicitly based the observed outcomes in array X, with no additional,
        unobserved outcomes. In combination with any estimator other than
        maximum likelihood, it may be useful to specify alphabets including
        unobserved outcomes. For such cases, successive possible outcomes of a
        random variable are indexed by the last axis in Alphabet_X; multiple
        alphabets may be specified using preceding axes, with the requirement
        X.shape[:-1]==Alphabet_X.shape[:-1]. Alphabets of different sizes may
        be specified either using numpy masked arrays, or by padding with the
        chosen placeholder fill_value.

        NB: When specifying multiple alphabets, an alphabet of possible joint
        outcomes is always implicit from the alphabets of possible (marginal)
        outcomes in Alphabet_X. For example, specifying
        Alphabet_X=np.array(((1,2),(1,2))) implies an alphabet of possible
        joint outcomes np.array(((1,1,2,2),(1,2,1,2))).
    keep_dims : boolean
        When set to True, an additional dimension of length one is appended to
        the returned array, facilitating any broadcast operations required by
        the user (defaults to False).

    **Implementation notes**:

    Before estimation, outcomes are mapped to the set of non-negative integers
    internally, with the value -1 representing missing data. To avoid this
    internal conversion step, supply integer data and use the default fill
    value -1.
    """
    W = information_binding(X, base, fill_value, estimator, Alphabet_X) + \
        information_multi(X, base, fill_value, estimator, Alphabet_X)

    if keep_dims:
        W = W[..., np.newaxis]

    return W


def information_enigmatic(X, base=2, fill_value=-1, estimator='ML',
                          Alphabet_X=None, keep_dims=False):
    # Note: can be negative
    # Note: equals multivariate mutual information when N=3, can test for this
    """
    Returns the estimated enigmatic information [JaEC11] for an array X
    containing realisations of discrete random variables.

    **Mathematical definition**:

    Given discrete random variables :math:`X_1, \ldots, X_n`, the enigmatic
    information :math:`Q(X_1, \\ldots, X_n)` is defined as:

    .. math::
        Q(X_1, \\ldots, X_n) = T(X_1, \\ldots, X_n) - B(X_1, \\ldots, X_n)

    where :math:`T(\\cdot, \\ldots, \\cdot)` denotes the multi-information and
    where :math:`B(\\cdot, \\ldots, \\cdot)` denotes the binding information.

    **Estimation**:

    Enigmatic information is estimated based on frequency tables, using the
    following functions:

        entropy_joint()

        entropy()

    See below for a list of available estimators. Note that although enigmatic
    information is a non-negative quantity, depending on the chosen estimator
    the obtained estimate may be negative.

    **Parameters**:

    X : numpy array (or array-like object such as a list of immutables, as \
    accepted by np.array())
        An array containing discrete random variable realisations. Successive
        realisations of a random variable are indexed by the last axis in the
        array; multiple random variables may be specified using preceding axes.
        When X.ndim==1, returns the scalar 0. When X.ndim>1, returns a scalar
        based on jointly considering all random variables indexed in the array.
        X may not contain (floating point) NaN values. Missing data may be
        specified using numpy masked arrays, as well as using standard
        numpy array/array-like objects; see below for details.
    base : float
        The desired logarithmic base (default 2).
    fill_value : object
        It is possible to specify missing data using numpy masked arrays,
        pandas Series/DataFrames, as well as using standard numpy
        array/array-like objects with assigned placeholder values. When using
        numpy masked arrays, this function invokes np.ma.filled() internally,
        so that missing data are represented with the array's object-internal
        placeholder value fill_value (this function's fill_value parameter is
        ignored in such cases). When using pandas Series/DataFrames, an initial
        conversion to a numpy masked array is performed. When using standard
        numpy array/array-like objects, this function's fill_value parameter is
        used to specify the placeholder value for missing data (defaults to
        -1).

        Data equal to the placeholder value are subsequently ignored.
    estimator : str or float
        The desired estimator (see above for details on estimators). Possible
        values are:

            *'ML' (the default value)* : Maximum likelihood estimator.

            *any floating point value* : Maximum a posteriori esimator using
            Dirichlet prior (equivalent to maximum likelihood with pseudo-count
            for each outcome as specified).

            *PERKS* : Maximum a posteriori esimator using Dirichlet prior
            (equivalent to maximum likelihood with pseudo-count for each
            outcome set to 1/L, where L is the number of possible outcomes.

            *MINIMAX* : Maximum a posteriori esimator using Dirichlet prior
            (equivalent to maximum likelihood with pseudo-count for each
            outcome set to sqrt(N)/L, where N is the total number of
            realisations and where L is the number of possible outcomes.

            *JAMES-STEIN* : James-Stein estimator [HaSt09].

            *GOOD-TURING* : Good-Turing estimator [GaSa95].

    Alphabet_X : numpy array (or array-like object such as a list of \
    immutables, as accepted by np.array())
        An array specifying the alphabet/alphabets of possible outcomes that
        random variable realisations in array X may assume. Defaults to None,
        in which case the alphabet/alphabets of possible outcomes is/are
        implicitly based the observed outcomes in array X, with no additional,
        unobserved outcomes. In combination with any estimator other than
        maximum likelihood, it may be useful to specify alphabets including
        unobserved outcomes. For such cases, successive possible outcomes of a
        random variable are indexed by the last axis in Alphabet_X; multiple
        alphabets may be specified using preceding axes, with the requirement
        X.shape[:-1]==Alphabet_X.shape[:-1]. Alphabets of different sizes may
        be specified either using numpy masked arrays, or by padding with the
        chosen placeholder fill_value.

        NB: When specifying multiple alphabets, an alphabet of possible joint
        outcomes is always implicit from the alphabets of possible (marginal)
        outcomes in Alphabet_X. For example, specifying
        Alphabet_X=np.array(((1,2),(1,2))) implies an alphabet of possible
        joint outcomes np.array(((1,1,2,2),(1,2,1,2))).
    keep_dims : boolean
        When set to True, an additional dimension of length one is appended to
        the returned array, facilitating any broadcast operations required by
        the user (defaults to False).

    **Implementation notes**:

    Before estimation, outcomes are mapped to the set of non-negative integers
    internally, with the value -1 representing missing data. To avoid this
    internal conversion step, supply integer data and use the default fill
    value -1.
    """
    Q = information_multi(X, base, fill_value, estimator, Alphabet_X) - \
        information_binding(X, base, fill_value, estimator, Alphabet_X)

    if keep_dims:
        Q = Q[..., np.newaxis]

    return Q


def information_interaction(X, base=2, fill_value=-1, estimator='ML',
                            Alphabet_X=None, keep_dims=False):
    """
    Returns the estimated interaction information [JaBr03] for an array X
    containing realisations of discrete random variables.

    **Mathematical definition**:

    Given discrete random variables :math:`X_1, \ldots, X_n`, the interaction
    information :math:`\\mathrm{Int}(X_1, \\ldots, X_n)` is defined as:

    .. math::
        \\mathrm{Int}(X_1, \\ldots, X_n) = - \\sum_{T \\subseteq
        \\{1,\\ldots, n\\}} (-1)^{n-|T|}  H(X_i : i \in T)

    where :math:`H(X_i : i \in T)` denotes the joint entropy of the subset of
    random variables specified by :math:`T`. Thus, interaction information is
    an alternating sum of joint entropies, with the sets of random variables
    used to compute the joint entropy in each term selected from the power set
    of available random variables.

    Note that interaction information is equal in magnitude to the
    co-information :math:`I(X_1, \\ldots, X_n)`, with equality for the case
    where :math:`n` is even,

    .. math::
        \\mathrm{Int}(X_1, \\ldots, X_n) = (-1)^n I(X_1, \\ldots, X_n).

    **Estimation**:

    Interaction information is estimated based on frequency tables, using the
    following functions:

        entropy_joint()

    See below for a list of available estimators. Note that although
    interaction information is a non-negative quantity, depending on the chosen
    estimator the obtained estimate may be negative.

    **Parameters**:

    X : numpy array (or array-like object such as a list of immutables, as \
    accepted by np.array())
        An array containing discrete random variable realisations. Successive
        realisations of a random variable are indexed by the last axis in the
        array; multiple random variables may be specified using preceding axes.
        When X.ndim==1, returns a scalar and is equivalent to -1*entropy().
        When X.ndim>1, returns a scalar based on jointly considering all random
        variables indexed in the array. X may not contain (floating point) NaN
        values. Missing data may be specified using numpy masked arrays, as
        well as using standard numpy array/array-like objects; see below
        for details.
    base : float
        The desired logarithmic base (default 2).
    fill_value : object
        It is possible to specify missing data using numpy masked arrays,
        pandas Series/DataFrames, as well as using standard numpy
        array/array-like objects with assigned placeholder values. When using
        numpy masked arrays, this function invokes np.ma.filled() internally,
        so that missing data are represented with the array's object-internal
        placeholder value fill_value (this function's fill_value parameter is
        ignored in such cases). When using pandas Series/DataFrames, an initial
        conversion to a numpy masked array is performed. When using standard
        numpy array/array-like objects, this function's fill_value parameter is
        used to specify the placeholder value for missing data (defaults to
        -1).

        Data equal to the placeholder value are subsequently ignored.
    estimator : str or float
        The desired estimator (see above for details on estimators). Possible
        values are:

            *'ML' (the default value)* : Maximum likelihood estimator.

            *any floating point value* : Maximum a posteriori esimator using
            Dirichlet prior (equivalent to maximum likelihood with pseudo-count
            for each outcome as specified).

            *PERKS* : Maximum a posteriori esimator using Dirichlet prior
            (equivalent to maximum likelihood with pseudo-count for each
            outcome set to 1/L, where L is the number of possible outcomes.

            *MINIMAX* : Maximum a posteriori esimator using Dirichlet prior
            (equivalent to maximum likelihood with pseudo-count for each
            outcome set to sqrt(N)/L, where N is the total number of
            realisations and where L is the number of possible outcomes.

            *JAMES-STEIN* : James-Stein estimator [HaSt09].

            *GOOD-TURING* : Good-Turing estimator [GaSa95].

    Alphabet_X : numpy array (or array-like object such as a list of \
    immutables, as accepted by np.array())
        An array specifying the alphabet/alphabets of possible outcomes that
        random variable realisations in array X may assume. Defaults to None,
        in which case the alphabet/alphabets of possible outcomes is/are
        implicitly based the observed outcomes in array X, with no additional,
        unobserved outcomes. In combination with any estimator other than
        maximum likelihood, it may be useful to specify alphabets including
        unobserved outcomes. For such cases, successive possible outcomes of a
        random variable are indexed by the last axis in Alphabet_X; multiple
        alphabets may be specified using preceding axes, with the requirement
        X.shape[:-1]==Alphabet_X.shape[:-1]. Alphabets of different sizes may
        be specified either using numpy masked arrays, or by padding with the
        chosen placeholder fill_value.

        NB: When specifying multiple alphabets, an alphabet of possible joint
        outcomes is always implicit from the alphabets of possible (marginal)
        outcomes in Alphabet_X. For example, specifying
        Alphabet_X=np.array(((1,2),(1,2))) implies an alphabet of possible
        joint outcomes np.array(((1,1,2,2),(1,2,1,2))).
    keep_dims : boolean
        When set to True, an additional dimension of length one is appended to
        the returned array, facilitating any broadcast operations required by
        the user (defaults to False).

    **Implementation notes**:

    Before estimation, outcomes are mapped to the set of non-negative integers
    internally, with the value -1 representing missing data. To avoid this
    internal conversion step, supply integer data and use the default fill
    value -1.
    """
    X, fill_value_X = _sanitise_array_input(X, fill_value)
    if Alphabet_X is not None:
        Alphabet_X, fill_value_Alphabet_X = _sanitise_array_input(Alphabet_X,
                                                                  fill_value)
        Alphabet_X, _ = _autocreate_alphabet(Alphabet_X, fill_value_Alphabet_X)
    else:
        Alphabet_X, fill_value_Alphabet_X = _autocreate_alphabet(X,
                                                                 fill_value_X)

    if X.size == 0:
        raise ValueError("arg X contains no elements")
    if np.any(_isnan(X)):
        raise ValueError("arg X contains NaN values")
    if Alphabet_X.size == 0:
        raise ValueError("arg Alphabet_X contains no elements")
    if np.any(_isnan(Alphabet_X)):
        raise ValueError("arg Alphabet_X contains NaN values")
    if _isnan(fill_value_X):
        raise ValueError("fill value for arg X is NaN")
    if X.shape[:-1] != Alphabet_X.shape[:-1]:
        raise ValueError("leading dimensions of args X and Alphabet_X do not "
                         "match")
    if not (np.isscalar(base) and np.isreal(base) and base > 0):
        raise ValueError("arg base not a positive real-valued scalar")

    S, fill_value = _map_observations_to_integers((X, Alphabet_X),
                                                  (fill_value_X,
                                                   fill_value_Alphabet_X))
    X, Alphabet_X = S

    # Re-shape X, so that we may handle multi-dimensional arrays equivalently
    # and iterate across 0th axis
    X = np.reshape(X, (-1, X.shape[-1]))
    Alphabet_X = np.reshape(Alphabet_X, (-1, Alphabet_X.shape[-1]))

    I = 0
    M = np.zeros(X.shape[0]).astype('bool')
    M = _increment_binary_vector(M)
    while np.any(M):
        I -= (-1)**(X.shape[0]-np.sum(M)) * \
            entropy_joint(X[M], base, fill_value, estimator, Alphabet_X[M])
        M = _increment_binary_vector(M)

    if keep_dims:
        I = I[..., np.newaxis]

    return I


def information_co(X, base=2, fill_value=-1, estimator='ML', Alphabet_X=None,
                   keep_dims=False):
    """
    Returns the estimated co-information [Bell03] for an array X containing
    realisations of discrete random variables.

    **Mathematical definition**:

    Given discrete random variables :math:`X_1, \ldots, X_n`, the
    co-information :math:`I(X_1, \\ldots, X_n)` is defined as:

    .. math::
        I(X_1, \\ldots, X_n) = - \\sum_{T \\subseteq \\{1,\\ldots, n\\}}
        (-1)^{|T|}  H(X_i : i \in T)

    where :math:`H(X_i : i \in T)` denotes the joint entropy of the subset of
    random variables specified by :math:`T`. Thus, co-information is an
    alternating sum of joint entropies, with the sets of random variables used
    to compute the joint entropy in each term selected from the power set of
    available random variables.

    Note that co-information is equal in magnitude to the interaction
    information :math:`\\mathrm{Int}(X_1, \\ldots, X_n)`, with equality for the
    case where :math:`n` is even,

    .. math::
        I(X_1, \\ldots, X_n) = (-1)^n \\mathrm{Int}(X_1, \\ldots, X_n).

    **Estimation**:

    Co-information is estimated based on frequency tables, using the following
    functions:

        entropy_joint()

    See below for a list of available estimators. Note that although
    co-information is a non-negative quantity, depending on the chosen
    estimator the obtained estimate may be negative.

    **Parameters**:

    X : numpy array (or array-like object such as a list of immutables, as \
    accepted by np.array())
        An array containing discrete random variable realisations. Successive
        realisations of a random variable are indexed by the last axis in the
        array; multiple random variables may be specified using preceding axes.
        When X.ndim==1, returns a scalar and is equivalent to entropy(). When
        X.ndim>1, returns a scalar based on jointly considering all random
        variables indexed in the array. X may not contain (floating point) NaN
        values. Missing data may be specified using numpy masked arrays, as
        well as using standard numpy array/array-like objects; see below
        for details.
    base : float
        The desired logarithmic base (default 2).
    fill_value : object
        It is possible to specify missing data using numpy masked arrays,
        pandas Series/DataFrames, as well as using standard numpy
        array/array-like objects with assigned placeholder values. When using
        numpy masked arrays, this function invokes np.ma.filled() internally,
        so that missing data are represented with the array's object-internal
        placeholder value fill_value (this function's fill_value parameter is
        ignored in such cases). When using pandas Series/DataFrames, an initial
        conversion to a numpy masked array is performed. When using standard
        numpy array/array-like objects, this function's fill_value parameter is
        used to specify the placeholder value for missing data (defaults to
        -1).

        Data equal to the placeholder value are subsequently ignored.
    estimator : str or float
        The desired estimator (see above for details on estimators). Possible
        values are:

            *'ML' (the default value)* : Maximum likelihood estimator.

            *any floating point value* : Maximum a posteriori esimator using
            Dirichlet prior (equivalent to maximum likelihood with pseudo-count
            for each outcome as specified).

            *PERKS* : Maximum a posteriori esimator using Dirichlet prior
            (equivalent to maximum likelihood with pseudo-count for each
            outcome set to 1/L, where L is the number of possible outcomes.

            *MINIMAX* : Maximum a posteriori esimator using Dirichlet prior
            (equivalent to maximum likelihood with pseudo-count for each
            outcome set to sqrt(N)/L, where N is the total number of
            realisations and where L is the number of possible outcomes.

            *JAMES-STEIN* : James-Stein estimator [HaSt09].

            *GOOD-TURING* : Good-Turing estimator [GaSa95].

    Alphabet_X : numpy array (or array-like object such as a list of \
    immutables, as accepted by np.array())
        An array specifying the alphabet/alphabets of possible outcomes that
        random variable realisations in array X may assume. Defaults to None,
        in which case the alphabet/alphabets of possible outcomes is/are
        implicitly based the observed outcomes in array X, with no additional,
        unobserved outcomes. In combination with any estimator other than
        maximum likelihood, it may be useful to specify alphabets including
        unobserved outcomes. For such cases, successive possible outcomes of a
        random variable are indexed by the last axis in Alphabet_X; multiple
        alphabets may be specified using preceding axes, with the requirement
        X.shape[:-1]==Alphabet_X.shape[:-1]. Alphabets of different sizes may
        be specified either using numpy masked arrays, or by padding with the
        chosen placeholder fill_value.

        NB: When specifying multiple alphabets, an alphabet of possible joint
        outcomes is always implicit from the alphabets of possible (marginal)
        outcomes in Alphabet_X. For example, specifying
        Alphabet_X=np.array(((1,2),(1,2))) implies an alphabet of possible
        joint outcomes np.array(((1,1,2,2),(1,2,1,2))).
    keep_dims : boolean
        When set to True, an additional dimension of length one is appended to
        the returned array, facilitating any broadcast operations required by
        the user (defaults to False).

    **Implementation notes**:

    Before estimation, outcomes are mapped to the set of non-negative integers
    internally, with the value -1 representing missing data. To avoid this
    internal conversion step, supply integer data and use the default fill
    value -1.

    """
    X, fill_value_X = _sanitise_array_input(X, fill_value)
    if Alphabet_X is not None:
        Alphabet_X, fill_value_Alphabet_X = \
            _sanitise_array_input(Alphabet_X, fill_value)
        Alphabet_X, _ = _autocreate_alphabet(Alphabet_X, fill_value_Alphabet_X)
    else:
        Alphabet_X, fill_value_Alphabet_X = \
            _autocreate_alphabet(X, fill_value_X)

    if X.size == 0:
        raise ValueError("arg X contains no elements")
    if np.any(_isnan(X)):
        raise ValueError("arg X contains NaN values")
    if Alphabet_X.size == 0:
        raise ValueError("arg Alphabet_X contains no elements")
    if np.any(_isnan(Alphabet_X)):
        raise ValueError("arg Alphabet_X contains NaN values")
    if _isnan(fill_value_X):
        raise ValueError("fill value for arg X is NaN")
    if X.shape[:-1] != Alphabet_X.shape[:-1]:
        raise ValueError("leading dimensions of args X and Alphabet_X do not "
                         "match")
    if not (np.isscalar(base) and np.isreal(base) and base > 0):
        raise ValueError("arg base not a positive real-valued scalar")

    S, fill_value = _map_observations_to_integers((X, Alphabet_X),
                                                  (fill_value_X,
                                                   fill_value_Alphabet_X))
    X, Alphabet_X = S

    # Re-shape X, so that we may handle multi-dimensional arrays equivalently
    # and iterate across 0th axis
    X = np.reshape(X, (-1, X.shape[-1]))
    Alphabet_X = np.reshape(Alphabet_X, (-1, Alphabet_X.shape[-1]))

    I = 0
    M = np.zeros(X.shape[0]).astype('bool')
    M = _increment_binary_vector(M)
    while np.any(M):
        I -= (-1)**(np.sum(M)) * \
            entropy_joint(X[M], base, fill_value, estimator, Alphabet_X[M])
        M = _increment_binary_vector(M)

    if keep_dims:
        I = I[..., np.newaxis]

    return I


def information_binding(X, base=2, fill_value=-1, estimator='ML',
                        Alphabet_X=None, keep_dims=False):
    """
    Returns the estimated binding information [AbPl12] (also known as dual
    total correlation [Han78]) for an array X containing realisations of
    discrete random variables.

    **Mathematical definition**:

    Given discrete random variables :math:`X_1, \ldots, X_n`, the binding
    information :math:`B(X_1, \\ldots, X_n)` is defined as:

    .. math::
        B(X_1, \\ldots, X_n) = H(X_1, \\ldots, X_n) -
        \\sum_{i=1}^{n} H(X_i | X_1, \\ldots X_{i-1}, X_{i+1}, \\ldots, X_n)

    where :math:`H(\\cdot)` denotes the entropy and where
    :math:`H(\\cdot | \\cdot)` denotes the conditional entropy.

    **Estimation**:

    Binding information is estimated based on frequency tables, using the
    following functions:

        entropy_joint()

        entropy()

    See below for a list of available estimators. Note that although binding
    information is a non-negative quantity, depending on the chosen estimator
    the obtained estimate may be negative.

    **Parameters**:

    X : numpy array (or array-like object such as a list of immutables, as \
    accepted by np.array())
        An array containing discrete random variable realisations. Successive
        realisations of a random variable are indexed by the last axis in the
        array; multiple random variables may be specified using preceding axes.
        When X.ndim==1, returns the scalar 0. When X.ndim>1, returns a scalar
        based on jointly considering all random variables indexed in the array.
        X may not contain (floating point) NaN values. Missing data may be
        specified using numpy masked arrays, as well as using standard
        numpy array/array-like objects; see below for details.
    base : float
        The desired logarithmic base (default 2).
    fill_value : object
        It is possible to specify missing data using numpy masked arrays,
        pandas Series/DataFrames, as well as using standard numpy
        array/array-like objects with assigned placeholder values. When using
        numpy masked arrays, this function invokes np.ma.filled() internally,
        so that missing data are represented with the array's object-internal
        placeholder value fill_value (this function's fill_value parameter is
        ignored in such cases). When using pandas Series/DataFrames, an initial
        conversion to a numpy masked array is performed. When using standard
        numpy array/array-like objects, this function's fill_value parameter is
        used to specify the placeholder value for missing data (defaults to
        -1).

        Data equal to the placeholder value are subsequently ignored.
    estimator : str or float
        The desired estimator (see above for details on estimators). Possible
        values are:

            *'ML' (the default value)* : Maximum likelihood estimator.

            *any floating point value* : Maximum a posteriori esimator using
            Dirichlet prior (equivalent to maximum likelihood with pseudo-count
            for each outcome as specified).

            *PERKS* : Maximum a posteriori esimator using Dirichlet prior
            (equivalent to maximum likelihood with pseudo-count for each
            outcome set to 1/L, where L is the number of possible outcomes.

            *MINIMAX* : Maximum a posteriori esimator using Dirichlet prior
            (equivalent to maximum likelihood with pseudo-count for each
            outcome set to sqrt(N)/L, where N is the total number of
            realisations and where L is the number of possible outcomes.

            *JAMES-STEIN* : James-Stein estimator [HaSt09].

            *GOOD-TURING* : Good-Turing estimator [GaSa95].

    Alphabet_X : numpy array (or array-like object such as a list of \
    immutables, as accepted by np.array())
        An array specifying the alphabet/alphabets of possible outcomes that
        random variable realisations in array X may assume. Defaults to None,
        in which case the alphabet/alphabets of possible outcomes is/are
        implicitly based the observed outcomes in array X, with no additional,
        unobserved outcomes. In combination with any estimator other than
        maximum likelihood, it may be useful to specify alphabets including
        unobserved outcomes. For such cases, successive possible outcomes of a
        random variable are indexed by the last axis in Alphabet_X; multiple
        alphabets may be specified using preceding axes, with the requirement
        X.shape[:-1]==Alphabet_X.shape[:-1]. Alphabets of different sizes may
        be specified either using numpy masked arrays, or by padding with the
        chosen placeholder fill_value.

        NB: When specifying multiple alphabets, an alphabet of possible joint
        outcomes is always implicit from the alphabets of possible (marginal)
        outcomes in Alphabet_X. For example, specifying
        Alphabet_X=np.array(((1,2),(1,2))) implies an alphabet of possible
        joint outcomes np.array(((1,1,2,2),(1,2,1,2))).
    keep_dims : boolean
        When set to True, an additional dimension of length one is appended to
        the returned array, facilitating any broadcast operations required by
        the user (defaults to False).

    **Implementation notes**:

    Before estimation, outcomes are mapped to the set of non-negative integers
    internally, with the value -1 representing missing data. To avoid this
    internal conversion step, supply integer data and use the default fill
    value -1.

    """
    X, fill_value_X = _sanitise_array_input(X, fill_value)
    if Alphabet_X is not None:
        Alphabet_X, fill_value_Alphabet_X = _sanitise_array_input(Alphabet_X,
                                                                  fill_value)
        Alphabet_X, _ = _autocreate_alphabet(Alphabet_X, fill_value_Alphabet_X)
    else:
        Alphabet_X, fill_value_Alphabet_X = _autocreate_alphabet(X,
                                                                 fill_value_X)

    # Exceptions needed to create Alphabet_X correctly if None
    # Not all of these are needed, however we include them for consistency.
    if X.size == 0:
        raise ValueError("arg X contains no elements")
    if np.any(_isnan(X)):
        raise ValueError("arg X contains NaN values")
    if Alphabet_X.size == 0:
        raise ValueError("arg Alphabet_X contains no elements")
    if np.any(_isnan(Alphabet_X)):
        raise ValueError("arg Alphabet_X contains NaN values")
    if _isnan(fill_value_X):
        raise ValueError("fill value for arg X is NaN")
    if X.shape[:-1] != Alphabet_X.shape[:-1]:
        raise ValueError("leading dimensions of args X and Alphabet_X do not "
                         "match")
    if not (np.isscalar(base) and np.isreal(base) and base > 0):
        raise ValueError("arg base not a positive real-valued scalar")

    S, fill_value = _map_observations_to_integers((X, Alphabet_X),
                                                  (fill_value_X,
                                                   fill_value_Alphabet_X))
    X, Alphabet_X = S

    # Exceptions thrown by entropy_joint
    H_joint = entropy_joint(X, base, fill_value, estimator, Alphabet_X)
    B = H_joint
    # Re-shape X, so that we may handle multi-dimensional arrays equivalently
    # and iterate across 0th axis
    X = np.reshape(X, (-1, X.shape[-1]))
    Alphabet_X = np.reshape(Alphabet_X, (-1, Alphabet_X.shape[-1]))
    M = np.arange(X.shape[0])

    for i in range(X.shape[0]):
        B -= H_joint
        if X.shape[0] > 1:
            B += entropy_joint(X[M != i], base, fill_value, estimator,
                               Alphabet_X[M != i])

    if keep_dims:
        B = B[..., np.newaxis]

    return B


def information_multi(X, base=2, fill_value=-1, estimator='ML',
                      Alphabet_X=None, keep_dims=False):
    """
    Returns the estimated multi-information [StVe98] (also known as total
    correlation [Wata60]) for an array X containing realisations of discrete
    random variables.

    **Mathematical definition**:

    Given discrete random variables :math:`X_1, \ldots, X_n`, the
    multi-information :math:`T(X_1, \\ldots, X_n)` is defined as:

    .. math::
        T(X_1, \\ldots, X_n) = \\left( \\sum_{i=1}^{n} H(X_i) \\right) -
        H(X_1, \\ldots, X_n)

    where :math:`H(\\cdot)` denotes the entropy and where
    :math:`H(\\cdot, \\ldots, \\cdot)` denotes the joint entropy.

    **Estimation**:

    Multi-information is estimated based on frequency tables, using the
    following functions:

        entropy_joint()

        entropy()

    See below for a list of available estimators. Note that although
    multi-information is a non-negative quantity, depending on the chosen
    estimator the obtained estimate may be negative.

    **Parameters**:

    X : numpy array (or array-like object such as a list of immutables, as \
    accepted by np.array())
        An array containing discrete random variable realisations. Successive
        realisations of a random variable are indexed by the last axis in the
        array; multiple random variables may be specified using preceding axes.
        When X.ndim==1, returns the scalar 0. When X.ndim>1, returns a scalar
        based on jointly considering all random variables indexed in the array.
        X may not contain (floating point) NaN values. Missing data may be
        specified using numpy masked arrays, as well as using standard
        numpy array/array-like objects; see below for details.
    base : float
        The desired logarithmic base (default 2).
    fill_value : object
        It is possible to specify missing data using numpy masked arrays,
        pandas Series/DataFrames, as well as using standard numpy
        array/array-like objects with assigned placeholder values. When using
        numpy masked arrays, this function invokes np.ma.filled() internally,
        so that missing data are represented with the array's object-internal
        placeholder value fill_value (this function's fill_value parameter is
        ignored in such cases). When using pandas Series/DataFrames, an initial
        conversion to a numpy masked array is performed. When using standard
        numpy array/array-like objects, this function's fill_value parameter is
        used to specify the placeholder value for missing data (defaults to
        -1).

        Data equal to the placeholder value are subsequently ignored.
    estimator : str or float
        The desired estimator (see above for details on estimators). Possible
        values are:

            *'ML' (the default value)* : Maximum likelihood estimator.

            *any floating point value* : Maximum a posteriori esimator using
            Dirichlet prior (equivalent to maximum likelihood with pseudo-count
            for each outcome as specified).

            *PERKS* : Maximum a posteriori esimator using Dirichlet prior
            (equivalent to maximum likelihood with pseudo-count for each
            outcome set to 1/L, where L is the number of possible outcomes.

            *MINIMAX* : Maximum a posteriori esimator using Dirichlet prior
            (equivalent to maximum likelihood with pseudo-count for each
            outcome set to sqrt(N)/L, where N is the total number of
            realisations and where L is the number of possible outcomes.

            *JAMES-STEIN* : James-Stein estimator [HaSt09].

            *GOOD-TURING* : Good-Turing estimator [GaSa95].

    Alphabet_X : numpy array (or array-like object such as a list of \
    immutables, as accepted by np.array())
        An array specifying the alphabet/alphabets of possible outcomes that
        random variable realisations in array X may assume. Defaults to None,
        in which case the alphabet/alphabets of possible outcomes is/are
        implicitly based the observed outcomes in array X, with no additional,
        unobserved outcomes. In combination with any estimator other than
        maximum likelihood, it may be useful to specify alphabets including
        unobserved outcomes. For such cases, successive possible outcomes of a
        random variable are indexed by the last axis in Alphabet_X; multiple
        alphabets may be specified using preceding axes, with the requirement
        X.shape[:-1]==Alphabet_X.shape[:-1]. Alphabets of different sizes may
        be specified either using numpy masked arrays, or by padding with the
        chosen placeholder fill_value.

        NB: When specifying multiple alphabets, an alphabet of possible joint
        outcomes is always implicit from the alphabets of possible (marginal)
        outcomes in Alphabet_X. For example, specifying
        Alphabet_X=np.array(((1,2),(1,2))) implies an alphabet of possible
        joint outcomes np.array(((1,1,2,2),(1,2,1,2))).
    keep_dims : boolean
        When set to True, an additional dimension of length one is appended to
        the returned array, facilitating any broadcast operations required by
        the user (defaults to False).

    **Implementation notes**:

    Before estimation, outcomes are mapped to the set of non-negative integers
    internally, with the value -1 representing missing data. To avoid this
    internal conversion step, supply integer data and use the default fill
    value -1.
    """
    H = entropy(X, base, fill_value, estimator, Alphabet_X)
    H_joint = entropy_joint(X, base, fill_value, estimator, Alphabet_X)

    T = np.sum(H) - H_joint

    if keep_dims:
        T = T[..., np.newaxis]

    return T


def information_mutual_conditional(X, Y, Z, cartesian_product=False, base=2,
                                   fill_value=-1, estimator='ML',
                                   Alphabet_X=None, Alphabet_Y=None,
                                   Alphabet_Z=None, keep_dims=False):
    """
    Returns the conditional mutual information (see e.g. [CoTh06]) between
    arrays X and Y given array Z, each containing discrete random variable
    realisations.

    **Mathematical definition**:

    Given discrete random variables :math:`X`, :math:`Y`,  :math:`Z`, the
    conditional mutual information :math:`I(X;Y|Z)` is defined as:

    .. math::
        I(X;Y|Z) = H(X|Z) - H(X|Y,Z)

    where :math:`H(\\cdot|\\cdot)` denotes the conditional entropy.

    **Estimation**:

    Conditional mutual information is estimated based on frequency tables,
    using the following functions:

        entropy_joint()

        entropy()

    See below for a list of available estimators. Note that although
    conditional mutual information is a non-negative quantity, depending on the
    chosen estimator the obtained estimate may be negative.

    **Parameters**:

    X,Y,Z : numpy array (or array-like object such as a list of immutables, \
    as accepted by np.array())
        *cartesian_product==False*: X,Y,Z are arrays containing discrete random
        variable realisations, with X.shape==Y.shape==Z.shape. Successive
        realisations of a random variable are indexed by the last axis in the
        respective arrays; multiple random variables in X,Y,Z may be specified
        using preceding axes of the respective arrays (random variables are
        paired **one-to-one** between X,Y,Z). When X.ndim==Y.ndim==Z.ndim==1,
        returns a scalar. When X.ndim>1 and Y.ndim>1 and Z.ndim>1, returns an
        array of estimated conditional mutual information values with
        dimensions X.shape[:-1]. Neither X nor Y nor Z may contain (floating
        point) NaN values. Missing data may be specified using numpy masked
        arrays, as well as using standard numpy array/array-like objects;
        see below for details.

        *cartesian_product==True*: X,Y,Z are arrays containing discrete random
        variable realisations, with X.shape[-1]==Y.shape[-1]==Z.shape[-1].
        Successive realisations of a random variable are indexed by the last
        axis in the respective arrays; multiple random variables in X,Y,Z may
        be specified using preceding axes of the respective arrays (random
        variables are paired **many-to-many** between X,Y,Z). When
        X.ndim==Y.ndim==Z.ndim==1, returns a scalar. When X.ndim>1 or Y.ndim>1
        or Z.ndim>1, returns an array of estimated conditional mutual
        information values with dimensions
        np.append(X.shape[:-1],Y.shape[:-1],Z.shape[:-1]). Neither X nor Y nor
        Z may contain (floating point) NaN values. Missing data may be
        specified using numpy masked arrays, as well as using standard
        numpy array/array-like objects; see below for details.
    cartesian_product : boolean
        Indicates whether random variables are paired **one-to-one** between
        X,Y,Z (cartesian_product==False, the default value) or **many-to-many**
        between X,Y,Z (cartesian_product==True).
    base : float
        The desired logarithmic base (default 2).
    fill_value : object
        It is possible to specify missing data using numpy masked arrays,
        pandas Series/DataFrames, as well as using standard numpy
        array/array-like objects with assigned placeholder values. When using
        numpy masked arrays, this function invokes np.ma.filled() internally,
        so that missing data are represented with the array's object-internal
        placeholder value fill_value (this function's fill_value parameter is
        ignored in such cases). When using pandas Series/DataFrames, an initial
        conversion to a numpy masked array is performed. When using standard
        numpy array/array-like objects, this function's fill_value parameter is
        used to specify the placeholder value for missing data (defaults to
        -1).

        Data equal to the placeholder value are subsequently ignored.
    estimator : str or float
        The desired estimator (see above for details on estimators). Possible
        values are:

            *'ML' (the default value)* : Maximum likelihood estimator.

            *any floating point value* : Maximum a posteriori esimator using
            Dirichlet prior (equivalent to maximum likelihood with pseudo-count
            for each outcome as specified).

            *PERKS* : Maximum a posteriori esimator using Dirichlet prior
            (equivalent to maximum likelihood with pseudo-count for each
            outcome set to 1/L, where L is the number of possible outcomes.

            *MINIMAX* : Maximum a posteriori esimator using Dirichlet prior
            (equivalent to maximum likelihood with pseudo-count for each
            outcome set to sqrt(N)/L, where N is the total number of
            realisations and where L is the number of possible outcomes.

            *JAMES-STEIN* : James-Stein estimator [HaSt09].

            *GOOD-TURING* : Good-Turing estimator [GaSa95].

    Alphabet_X, Alphabet_Y, Alphabet_Z : numpy array (or array-like object \
    such as a list of immutables, as accepted by np.array())
        Respectively an array specifying the alphabet/alphabets of possible
        outcomes that random variable realisations in array X, Y, Z may assume.
        Defaults to None, in which case the alphabet/alphabets of possible
        outcomes is/are implicitly based the observed outcomes in array X, Y, Z
        respectively, with no additional, unobserved outcomes. In combination
        with any estimator other than maximum likelihood, it may be useful to
        specify alphabets including unobserved outcomes. For such cases,
        successive possible outcomes of a random variable are indexed by the
        last axis in Alphabet_X, Alphabet_Y, Alphabet_Z respectively; multiple
        alphabets may be specified using preceding axes, with the requirement
        X.shape[:-1]==Alphabet_X.shape[:-1] (analogously for Y and Z).
        Alphabets of different sizes may be specified either using numpy masked
        arrays, or by padding with the chosen placeholder fill_value.

        NB: When specifying alphabets, an alphabet of possible joint outcomes
        is always implicit from the alphabets of possible (marginal) outcomes
        in Alphabet_X, Alphabet_Y, Alphabet_Z. For example, specifying
        Alphabet_X=Alphabet_Y=Alphabet_Z=np.array(((1,2)) implies an alphabet
        of possible joint outcomes
        np.array((1,1,1,1,2,2,2,2),((1,1,2,2,1,1,2,2),(1,2,1,2,1,2,1,2))).
    keep_dims : boolean
        When set to True and cartesian_product==False an additional dimension
        of length one is appended to the returned array, facilitating any
        broadcast operations required by the user (defaults to False). Has no
        effect when cartesian_product==True.

    **Implementation notes**:

    Before estimation, outcomes are mapped to the set of non-negative integers
    internally, with the value -1 representing missing data. To avoid this
    internal conversion step, supply integer data and use the default fill
    value -1.
    """
    X, fill_value_X = _sanitise_array_input(X, fill_value)
    Y, fill_value_Y = _sanitise_array_input(Y, fill_value)
    Z, fill_value_Z = _sanitise_array_input(Z, fill_value)
    if Alphabet_X is not None:
        Alphabet_X, fill_value_Alphabet_X = _sanitise_array_input(Alphabet_X,
                                                                  fill_value)
        Alphabet_X, _ = _autocreate_alphabet(Alphabet_X, fill_value_Alphabet_X)
    else:
        Alphabet_X, fill_value_Alphabet_X = _autocreate_alphabet(X,
                                                                 fill_value_X)
    if Alphabet_Y is not None:
        Alphabet_Y, fill_value_Alphabet_Y = _sanitise_array_input(Alphabet_Y,
                                                                  fill_value)
        Alphabet_Y, _ = _autocreate_alphabet(Alphabet_Y, fill_value_Alphabet_Y)
    else:
        Alphabet_Y, fill_value_Alphabet_Y = _autocreate_alphabet(Y,
                                                                 fill_value_Y)
    if Alphabet_Z is not None:
        Alphabet_Z, fill_value_Alphabet_Z = _sanitise_array_input(Alphabet_Z,
                                                                  fill_value)
        Alphabet_Z, _ = _autocreate_alphabet(Alphabet_Z, fill_value_Alphabet_Z)
    else:
        Alphabet_Z, fill_value_Alphabet_Z = _autocreate_alphabet(Z,
                                                                 fill_value_Z)

    if X.size == 0:
        raise ValueError("arg X contains no elements")
    if Y.size == 0:
        raise ValueError("arg Y contains no elements")
    if Z.size == 0:
        raise ValueError("arg Z contains no elements")
    if np.any(_isnan(X)):
        raise ValueError("arg X contains NaN values")
    if np.any(_isnan(Y)):
        raise ValueError("arg Y contains NaN values")
    if np.any(_isnan(Z)):
        raise ValueError("arg Z contains NaN values")
    if Alphabet_X.size == 0:
        raise ValueError("arg Alphabet_X contains no elements")
    if np.any(_isnan(Alphabet_X)):
        raise ValueError("arg Alphabet_X contains NaN values")
    if Alphabet_Y.size == 0:
        raise ValueError("arg Alphabet_Y contains no elements")
    if np.any(_isnan(Alphabet_Y)):
        raise ValueError("arg Alphabet_Y contains NaN values")
    if Alphabet_Z.size == 0:
        raise ValueError("arg Alphabet_Z contains no elements")
    if np.any(_isnan(Alphabet_Z)):
        raise ValueError("arg Alphabet_Z contains NaN values")
    if _isnan(fill_value_X):
        raise ValueError("fill value for arg X is NaN")
    if _isnan(fill_value_Y):
        raise ValueError("fill value for arg Y is NaN")
    if _isnan(fill_value_Z):
        raise ValueError("fill value for arg Z is NaN")
    if X.shape[:-1] != Alphabet_X.shape[:-1]:
        raise ValueError("leading dimensions of args X and Alphabet_X do not "
                         "match")
    if Y.shape[:-1] != Alphabet_Y.shape[:-1]:
        raise ValueError("leading dimensions of args Y and Alphabet_Y do not "
                         "match")
    if Z.shape[:-1] != Alphabet_Z.shape[:-1]:
        raise ValueError("leading dimensions of args Z and Alphabet_Z do not "
                         "match")
    if not cartesian_product and (X.shape != Y.shape or X.shape != Z.shape):
        raise ValueError("dimensions of args X, Y, Z do not match")
    if cartesian_product and (X.shape[-1] != Y.shape[-1] or
                              X.shape[-1] != Z.shape[-1]):
        raise ValueError("trailing dimensions of args X, Y, Z do not match")
    if not (np.isscalar(base) and np.isreal(base) and base > 0):
        raise ValueError("arg base not a positive real-valued scalar")

    S, fill_value = _map_observations_to_integers((X, Alphabet_X,
                                                   Y, Alphabet_Y,
                                                   Z, Alphabet_Z),
                                                  (fill_value_X,
                                                   fill_value_Alphabet_X,
                                                   fill_value_Y,
                                                   fill_value_Alphabet_Y,
                                                   fill_value_Z,
                                                   fill_value_Alphabet_Z))
    X, Alphabet_X, Y, Alphabet_Y, Z, Alphabet_Z = S

    if not cartesian_product:
        I = np.empty(X.shape[:-1])
        if I.ndim > 0:
            I[:] = np.NaN
        else:
            I = np.float64(np.NaN)
    else:
        shapeI_Z = Z.shape[:-1]
        Z = np.reshape(Z, (-1, Z.shape[-1]))
        Alphabet_Z = np.reshape(Alphabet_Z, (-1, Alphabet_Z.shape[-1]))
        I = []
        for i in range(Z.shape[0]):
            def f(X, Y, Alphabet_X, Alphabet_Y):
                return information_mutual_conditional(X, Y, Z[i], False, base,
                                                      fill_value, estimator,
                                                      Alphabet_X, Alphabet_Y,
                                                      Alphabet_Z[i])
            I.append(_cartesian_product_apply(X, Y, f, Alphabet_X, Alphabet_Y))
        shapeI_XY = I[0].shape
        if len(shapeI_Z) == 0:
            I = np.array(I)[0].reshape(shapeI_XY)
        else:
            I = np.array(I)
            I = np.rollaxis(I, 0, len(I.shape))
            I = I.reshape(np.append(shapeI_XY, shapeI_Z).astype('int'))
        return I

    # Re-shape H, X,Y,Z so that we may handle multi-dimensional arrays
    # equivalently and iterate across 0th axis
    X = np.reshape(X, (-1, X.shape[-1]))
    Y = np.reshape(Y, (-1, Y.shape[-1]))
    Z = np.reshape(Z, (-1, Z.shape[-1]))
    Alphabet_X = np.reshape(Alphabet_X, (-1, Alphabet_X.shape[-1]))
    Alphabet_Y = np.reshape(Alphabet_Y, (-1, Alphabet_Y.shape[-1]))
    Alphabet_Z = np.reshape(Alphabet_Z, (-1, Alphabet_Z.shape[-1]))
    orig_shape_I = I.shape
    I = np.reshape(I, (-1, 1))

    for i in range(X.shape[0]):
        I_ = (entropy_joint(np.vstack((X[i], Z[i])), base, fill_value,
                            estimator,
                            _vstack_pad((Alphabet_X[i],
                                         Alphabet_Z[i]),
                                        fill_value)) +
              entropy_joint(np.vstack((Y[i], Z[i])), base, fill_value,
                            estimator,
                            _vstack_pad((Alphabet_Y[i],
                                         Alphabet_Z[i]),
                                        fill_value)) -
              entropy_joint(np.vstack((X[i], Y[i], Z[i])), base, fill_value,
                            estimator,
                            _vstack_pad((Alphabet_X[i],
                                         Alphabet_Y[i],
                                         Alphabet_Z[i]), fill_value)) -
              entropy_joint(Z[i], base, fill_value, estimator, Alphabet_Z[i]))
        I[i] = I_

    # Reverse re-shaping
    I = np.reshape(I, orig_shape_I)

    if keep_dims and not cartesian_product:
        I = I[..., np.newaxis]

    return I


def information_lautum(X, Y=None, cartesian_product=False, base=2,
                       fill_value=-1, estimator='ML', Alphabet_X=None,
                       Alphabet_Y=None, keep_dims=False):
    """
    Returns the lautum information [PaVe08] between arrays X and Y, each
    containing discrete random variable realisations.

    **Mathematical definition**:

    Denoting with :math:`P_X(x)`, :math:`P_Y(x)` respectively the probability
    of observing an outcome :math:`x` with discrete random variables :math:`X`,
    :math:`Y`, and denoting with :math:`P_{XY}(x,y)` the probability of jointly
    observing outcomes :math:`x`, :math:`y` respectively with :math:`X`,
    :math:`Y`, the lautum information :math:`L(X;Y)` is defined as:

    .. math::
        \\begin{eqnarray}
            L(X;Y) &=& -\\sum_x \\sum_y
            {P_X(x) P_Y(y) \\log {\\frac{P_X(x) P_Y(y)}{P_{XY}(x,y)}}} \\\\
            &=& D_{\\mathrm{KL}}(P_X P_Y \\parallel P_{XY})
        \\end{eqnarray}

    where :math:`D_{\\mathrm{KL}}(\\cdot \\parallel \\cdot)` denotes the
    Kullback-Leibler divergence. Note that *lautum* is *mutual* spelt
    backwards; denoting with :math:`I(\\cdot;\\cdot)` the mutual information it
    may be shown (see e.g. [CoTh06]) that

    .. math::
        \\begin{eqnarray}
            I(X;Y) &=& D_{\\mathrm{KL}}(P_{XY} \\parallel P_X P_Y).
        \\end{eqnarray}

    **Estimation**:

    Lautum information is estimated based on frequency tables. See below for a
    list of available estimators.

    **Parameters**:

    X,Y : numpy array (or array-like object such as a list of immutables, as \
    accepted by np.array())
        *cartesian_product==False and Y is not None*: X and Y are arrays
        containing discrete random variable realisations, with
        X.shape[:-1]==Y.shape[:-1]. Successive realisations of a random
        variable are indexed by the last axis in the respective arrays;
        multiple random variables in X and Y may be specified using preceding
        axes of the respective arrays (random variables are paired
        **one-to-one** between X and Y). When X.ndim==Y.ndim==1, returns a
        scalar. When X.ndim>1 and Y.ndim>1, returns an array of estimated
        information values with dimensions X.shape[:-1]. Neither X nor Y may
        contain (floating point) NaN values. Missing data may be specified
        using numpy masked arrays, as well as using standard numpy
        array/array-like objects; see below for details.

        *cartesian_product==True and Y is not None*: X and Y are arrays
        containing discrete random variable realisations. Successive
        realisations of a random variable are indexed by the last axis in the
        respective arrays; multiple random variables in X and Y may be
        specified using preceding axes of the respective arrays (random
        variables are paired **many-to-many** between X and Y). When
        X.ndim==Y.ndim==1, returns a scalar. When X.ndim>1 or Y.ndim>1, returns
        an array of estimated information values with dimensions
        np.append(X.shape[:-1],Y.shape[:-1]). Neither X nor Y may contain
        (floating point) NaN values. Missing data may be specified using numpy
        masked arrays, as well as using standard numpy array/array-like
        objects; see below for details.

        *Y is None*: Equivalent to information_lautum(X, X, ... ). Thus, a
        shorthand syntax for computing lautum information (in bits) between all
        pairs of random variables in X is information_lautum(X).
    cartesian_product : boolean
        Indicates whether random variables are paired **one-to-one** between X
        and Y (cartesian_product==False, the default value) or **many-to-many**
        between X and Y (cartesian_product==True).
    base : float
        The desired logarithmic base (default 2).
    fill_value : object
        It is possible to specify missing data using numpy masked arrays,
        pandas Series/DataFrames, as well as using standard numpy
        array/array-like objects with assigned placeholder values. When using
        numpy masked arrays, this function invokes np.ma.filled() internally,
        so that missing data are represented with the array's object-internal
        placeholder value fill_value (this function's fill_value parameter is
        ignored in such cases). When using pandas Series/DataFrames, an initial
        conversion to a numpy masked array is performed. When using standard
        numpy array/array-like objects, this function's fill_value parameter is
        used to specify the placeholder value for missing data (defaults to
        -1).

        Data equal to the placeholder value are subsequently ignored.
    estimator : str or float
        The desired estimator (see above for details on estimators). Possible
        values are:

            *'ML' (the default value)* : Maximum likelihood estimator.

            *any floating point value* : Maximum a posteriori esimator using
            Dirichlet prior (equivalent to maximum likelihood with pseudo-count
            for each outcome as specified).

            *PERKS* : Maximum a posteriori esimator using Dirichlet prior
            (equivalent to maximum likelihood with pseudo-count for each
            outcome set to 1/L, where L is the number of possible outcomes.

            *MINIMAX* : Maximum a posteriori esimator using Dirichlet prior
            (equivalent to maximum likelihood with pseudo-count for each
            outcome set to sqrt(N)/L, where N is the total number of
            realisations and where L is the number of possible outcomes.

            *JAMES-STEIN* : James-Stein estimator [HaSt09].

            *GOOD-TURING* : Good-Turing estimator [GaSa95].

    Alphabet_X, Alphabet_Y : numpy array (or array-like object such as a list \
    of immutables, as accepted by np.array())
        Respectively an array specifying the alphabet/alphabets of possible
        outcomes that random variable realisations in array X, Y may assume.
        Defaults to None, in which case the alphabet/alphabets of possible
        outcomes is/are implicitly based the observed outcomes in array X, Y
        respectively, with no additional, unobserved outcomes. In combination
        with any estimator other than maximum likelihood, it may be useful to
        specify alphabets including unobserved outcomes. For such cases,
        successive possible outcomes of a random variable are indexed by the
        last axis in Alphabet_X, Alphabet_Y respectively; multiple alphabets
        may be specified using preceding axes, with the requirement
        X.shape[:-1]==Alphabet_X.shape[:-1] (analogously for Y). Alphabets of
        different sizes may be specified either using numpy masked arrays, or
        by padding with the chosen placeholder fill_value.

        NB: When specifying alphabets, an alphabet of possible joint outcomes
        is always implicit from the alphabets of possible (marginal) outcomes
        in Alphabet_X, Alphabet_Y. For example, specifying
        Alphabet_X=Alphabet_Y=np.array(((1,2)) implies an alphabet of possible
        joint outcomes np.array(((1,1,2,2),(1,2,1,2))).
    keep_dims : boolean
        When set to True and cartesian_product==False an additional dimension
        of length one is appended to the returned array, facilitating any
        broadcast operations required by the user (defaults to False). Has no
        effect when cartesian_product==True.

    **Implementation notes**:

    Before estimation, outcomes are mapped to the set of non-negative integers
    internally, with the value -1 representing missing data. To avoid this
    internal conversion step, supply integer data and use the default fill
    value -1.
    """
    if Y is None:
        Y = X
        cartesian_product = True
        Alphabet_Y = Alphabet_X

    X, fill_value_X = _sanitise_array_input(X, fill_value)
    Y, fill_value_Y = _sanitise_array_input(Y, fill_value)
    if Alphabet_X is not None:
        Alphabet_X, fill_value_Alphabet_X = _sanitise_array_input(Alphabet_X,
                                                                  fill_value)
        Alphabet_X, _ = _autocreate_alphabet(Alphabet_X, fill_value_Alphabet_X)
    else:
        Alphabet_X, fill_value_Alphabet_X = _autocreate_alphabet(X,
                                                                 fill_value_X)
    if Alphabet_Y is not None:
        Alphabet_Y, fill_value_Alphabet_Y = _sanitise_array_input(Alphabet_Y,
                                                                  fill_value)
        Alphabet_Y, _ = _autocreate_alphabet(Alphabet_Y, fill_value_Alphabet_Y)
    else:
        Alphabet_Y, fill_value_Alphabet_Y = _autocreate_alphabet(Y,
                                                                 fill_value_Y)

    if X.size == 0:
        raise ValueError("arg X contains no elements")
    if np.any(_isnan(X)):
        raise ValueError("arg X contains NaN values")
    if Y.size == 0:
        raise ValueError("arg Y contains no elements")
    if np.any(_isnan(Y)):
        raise ValueError("arg Y contains NaN values")
    if Alphabet_X.size == 0:
        raise ValueError("arg Alphabet_X contains no elements")
    if np.any(_isnan(Alphabet_X)):
        raise ValueError("arg Alphabet_X contains NaN values")
    if Alphabet_Y.size == 0:
        raise ValueError("arg Alphabet_Y contains no elements")
    if np.any(_isnan(Alphabet_Y)):
        raise ValueError("arg Alphabet_Y contains NaN values")
    if _isnan(fill_value_X):
        raise ValueError("fill value for arg X is NaN")
    if _isnan(fill_value_Y):
        raise ValueError("fill value for arg Y is NaN")
    if X.shape[:-1] != Alphabet_X.shape[:-1]:
        raise ValueError("leading dimensions of args X and Alphabet_X do not "
                         "match")
    if Y.shape[:-1] != Alphabet_Y.shape[:-1]:
        raise ValueError("leading dimensions of args Y and Alphabet_Y do not "
                         "match")
    if not cartesian_product and X.shape != Y.shape:
        raise ValueError("dimensions of args X and Y do not match")
    if cartesian_product and X.shape[-1] != Y.shape[-1]:
        raise ValueError("trailing dimensions of args X and Y do not match")
    if not (np.isscalar(base) and np.isreal(base) and base > 0):
        raise ValueError("arg base not a positive real-valued scalar")

    S, fill_value = _map_observations_to_integers((X, Alphabet_X,
                                                   Y, Alphabet_Y),
                                                  (fill_value_X,
                                                   fill_value_Alphabet_X,
                                                   fill_value_Y,
                                                   fill_value_Alphabet_Y))
    X, Alphabet_X, Y, Alphabet_Y = S

    if not cartesian_product:
        H = np.empty(X.shape[:-1])
        if H.ndim > 0:
            H[:] = np.NaN
        else:
            H = np.float64(np.NaN)
    else:
        def f(X, Y, Alphabet_X, Alphabet_Y):
            return information_lautum(X, Y, False, base, fill_value, estimator,
                                      Alphabet_X, Alphabet_Y)
        return _cartesian_product_apply(X, Y, f, Alphabet_X, Alphabet_Y)

    # Re-shape H, X and Y, so that we may handle multi-dimensional arrays
    # equivalently and iterate across 0th axis
    X = np.reshape(X, (-1, X.shape[-1]))
    Y = np.reshape(Y, (-1, Y.shape[-1]))
    Alphabet_X = np.reshape(Alphabet_X, (-1, Alphabet_X.shape[-1]))
    Alphabet_Y = np.reshape(Alphabet_Y, (-1, Alphabet_Y.shape[-1]))
    orig_shape_H = H.shape
    H = np.reshape(H, (-1, 1))

    _verify_alphabet_sufficiently_large(X, Alphabet_X, fill_value)
    _verify_alphabet_sufficiently_large(Y, Alphabet_Y, fill_value)

    for i in range(X.shape[0]):
        # Sort X and Y jointly, so that we can determine joint symbol
        # probabilities
        JointXY = np.vstack((X[i], Y[i]))
        JointXY = JointXY[:, JointXY[0].argsort(kind='mergesort')]
        JointXY = JointXY[:, JointXY[1].argsort(kind='mergesort')]

        # Compute symbol run-lengths
        # Compute symbol change indicators
        B = np.any(JointXY[:, 1:] != JointXY[:, :-1], axis=0)

        # Obtain symbol change positions
        I = np.append(np.where(B), JointXY.shape[1]-1)
        # Compute run lengths
        L = np.diff(np.append(-1, I))

        alphabet_XY = JointXY[:, I]
        if estimator != 'ML':
            L, alphabet_XY = \
                _append_empty_bins_using_alphabet(L, alphabet_XY,
                                                  _vstack_pad((Alphabet_X[i],
                                                               Alphabet_Y[i]),
                                                              fill_value),
                                                  fill_value)
        L, alphabet_XY = _remove_counts_at_fill_value(L, alphabet_XY,
                                                      fill_value)
        if not np.any(L):
            continue
        P_XY, _ = _estimate_probabilities(L, estimator)

        # Assign probabilities in P_XY to P_XY_reshaped, a matrix which
        # exhaustively records probabilities for all elements in the cartesian
        # product of alphabets. In this way, we can subsequently integrate
        # across variables X, Y.
        alphabet_X = np.unique(alphabet_XY[0])
        alphabet_Y = np.unique(alphabet_XY[1])
        P_XY_reshaped = np.zeros((alphabet_Y.size, alphabet_X.size))
        j = k = c = 0
        for c in range(P_XY.size):
            if c > 0 and alphabet_XY[1, c] != alphabet_XY[1, c-1]:
                k = 0
            while alphabet_X[k] != alphabet_XY[0, c]:
                k = k + 1
            while alphabet_Y[j] != alphabet_XY[1, c]:
                j = j + 1
            P_XY_reshaped[j, k] = P_XY[c]

        # Integrate across X, Y
        P_X = np.reshape(np.sum(P_XY_reshaped, axis=0), (1, -1))
        P_Y = np.reshape(np.sum(P_XY_reshaped, axis=1), (-1, 1))

        H[i] = divergence_kullbackleibler_pmf(np.reshape(P_X*P_Y,
                                                         (1, -1)),
                                              np.reshape(P_XY_reshaped,
                                                         (1, -1)),
                                              False, base)

    # Reverse re-shaping
    H = np.reshape(H, orig_shape_H)

    if keep_dims and not cartesian_product:
        H = H[..., np.newaxis]

    return H


def information_mutual_normalised(X, Y=None, norm_factor='Y',
                                  cartesian_product=False, fill_value=-1,
                                  estimator='ML', Alphabet_X=None,
                                  Alphabet_Y=None, keep_dims=False):
    # TODO Documentation should include properties for each of the
    # normalisation factors
    """
    Returns the normalised mutual information between arrays X and Y, each
    containing discrete random variable realisations.

    **Mathematical definition**:

    Given discrete random variables :math:`X`, :math:`Y`, the normalised mutual
    information :math:`NI(X;Y)` is defined as:

    .. math::
        NI(X;Y) = \\frac{I(X;Y)}{C_n}

    where :math:`I` denotes the mutual information and where :math:`C_n`
    denotes a normalisation factor. Normalised mutual information is a
    dimensionless quantity, with :math:`C_n` alternatively defined as:

    .. math::
         \\begin{eqnarray}
           C_{\\text{X}} &=& H(X) \\\\
           C_{\\text{Y}} &=& H(Y) \\\\
           C_{\\text{X+Y}} &=& H(X) + H(Y) \\\\
           C_{\\text{MIN}} &=& \\min \{ H(X), H(Y) \} \\\\
           C_{\\text{MAX}} &=& \\max \{ H(X), H(Y) \} \\\\
           C_{\\text{XY}} &=& H(X,Y) \\\\
           C_{\\text{SQRT}} &=& \\sqrt{H(X) H(Y)}
         \\end{eqnarray}

    where :math:`H(\\cdot)` and :math:`H(\\cdot,\\cdot)` respectively denote
    the entropy and joint entropy.

    **Estimation**:

    Normalised mutual information is estimated based on frequency tables, using
    the following functions:

        entropy_joint()

        entropy()

    See below for a list of available estimators. Note that although normalised
    mutual information is a non-negative quantity, depending on the chosen
    estimator the obtained estimate may be negative.

    **Parameters**:

    X,Y : numpy array (or array-like object such as a list of immutables, as \
    accepted by np.array())
        *cartesian_product==False and Y is not None*: X and Y are arrays
        containing discrete random variable realisations, with
        X.shape==Y.shape. Successive realisations of a random variable are
        indexed by the last axis in the respective arrays; multiple random
        variables in X and Y may be specified using preceding axes of the
        respective arrays (random variables are paired **one-to-one** between X
        and Y). When X.ndim==Y.ndim==1, returns a scalar. When X.ndim>1 and
        Y.ndim>1, returns an array of estimated normalised information values
        with dimensions X.shape[:-1]. Neither X nor Y may contain (floating
        point) NaN values. Missing data may be specified using numpy masked
        arrays, as well as using standard numpy array/array-like objects;
        see below for details.

        *cartesian_product==True and Y is not None*: X and Y are arrays
        containing discrete random variable realisations, with
        X.shape[-1]==Y.shape[-1]. Successive realisations of a random variable
        are indexed by the last axis in the respective arrays; multiple random
        variables in X and Y may be specified using preceding axes of the
        respective arrays (random variables are paired **many-to-many** between
        X and Y). When X.ndim==Y.ndim==1, returns a scalar. When X.ndim>1 or
        Y.ndim>1, returns an array of estimated normalised information values
        with dimensions np.append(X.shape[:-1],Y.shape[:-1]). Neither X nor Y
        may contain (floating point) NaN values. Missing data may be specified
        using numpy masked arrays, as well as using standard numpy
        array/array-like objects; see below for details.

        *Y is None*: Equivalent to information_mutual_normalised(X, X,
        norm_factor, True). Thus, a shorthand syntax for computing normalised
        mutual information (based on C_n = C_Y as defined above) between all
        pairs of random variables in X is information_mutual_normalised(X).
    norm_factor : string
        The desired normalisation factor, specified as a string. Internally,
        the supplied string is converted to upper case and spaces are
        discarded. Subsequently, the function tests for one of the following
        string values, each corresponding to an alternative normalisation
        factor as defined above:

        *'X'*

        *'Y' (the default value)*

        *'X+Y' (equivalently 'Y+X')*

        *'MIN'*

        *'MAX'*

        *'XY' (equivalently YX)*

        *'SQRT'*
    cartesian_product : boolean
        Indicates whether random variables are paired **one-to-one** between X
        and Y (cartesian_product==False, the default value) or **many-to-many**
        between X and Y (cartesian_product==True).
    fill_value : object
        It is possible to specify missing data using numpy masked arrays,
        pandas Series/DataFrames, as well as using standard numpy
        array/array-like objects with assigned placeholder values. When using
        numpy masked arrays, this function invokes np.ma.filled() internally,
        so that missing data are represented with the array's object-internal
        placeholder value fill_value (this function's fill_value parameter is
        ignored in such cases). When using pandas Series/DataFrames, an initial
        conversion to a numpy masked array is performed. When using standard
        numpy array/array-like objects, this function's fill_value parameter is
        used to specify the placeholder value for missing data (defaults to
        -1).

        Data equal to the placeholder value are subsequently ignored.
    estimator : str or float
        The desired estimator (see above for details on estimators). Possible
        values are:

            *'ML' (the default value)* : Maximum likelihood estimator.

            *any floating point value* : Maximum a posteriori esimator using
            Dirichlet prior (equivalent to maximum likelihood with pseudo-count
            for each outcome as specified).

            *PERKS* : Maximum a posteriori esimator using Dirichlet prior
            (equivalent to maximum likelihood with pseudo-count for each
            outcome set to 1/L, where L is the number of possible outcomes.

            *MINIMAX* : Maximum a posteriori esimator using Dirichlet prior
            (equivalent to maximum likelihood with pseudo-count for each
            outcome set to sqrt(N)/L, where N is the total number of
            realisations and where L is the number of possible outcomes.

            *JAMES-STEIN* : James-Stein estimator [HaSt09].

            *GOOD-TURING* : Good-Turing estimator [GaSa95].

    Alphabet_X, Alphabet_Y : numpy array (or array-like object such as a list \
    of immutables, as accepted by np.array())
        Respectively an array specifying the alphabet/alphabets of possible
        outcomes that random variable realisations in array X, Y may assume.
        Defaults to None, in which case the alphabet/alphabets of possible
        outcomes is/are implicitly based the observed outcomes in array X, Y
        respectively, with no additional, unobserved outcomes. In combination
        with any estimator other than maximum likelihood, it may be useful to
        specify alphabets including unobserved outcomes. For such cases,
        successive possible outcomes of a random variable are indexed by the
        last axis in Alphabet_X, Alphabet_Y respectively; multiple alphabets
        may be specified using preceding axes, with the requirement
        X.shape[:-1]==Alphabet_X.shape[:-1] (analogously for Y). Alphabets of
        different sizes may be specified either using numpy masked arrays, or
        by padding with the chosen placeholder fill_value.

        NB: When specifying alphabets, an alphabet of possible joint outcomes
        is always implicit from the alphabets of possible (marginal) outcomes
        in Alphabet_X, Alphabet_Y. For example, specifying
        Alphabet_X=Alphabet_Y=np.array(((1,2)) implies an alphabet of possible
        joint outcomes np.array(((1,1,2,2),(1,2,1,2))).
    keep_dims : boolean
        When set to True and cartesian_product==False an additional dimension
        of length one is appended to the returned array, facilitating any
        broadcast operations required by the user (defaults to False). Has no
        effect when cartesian_product==True.

    **Implementation notes**:

    Before estimation, outcomes are mapped to the set of non-negative integers
    internally, with the value -1 representing missing data. To avoid this
    internal conversion step, supply integer data and use the default fill
    value -1.
    """
    if Y is None:
        Y = X
        cartesian_product = True
        Alphabet_Y = Alphabet_X

    X, fill_value_X = _sanitise_array_input(X, fill_value)
    Y, fill_value_Y = _sanitise_array_input(Y, fill_value)
    if Alphabet_X is not None:
        Alphabet_X, fill_value_Alphabet_X = _sanitise_array_input(Alphabet_X,
                                                                  fill_value)
        Alphabet_X, _ = _autocreate_alphabet(Alphabet_X, fill_value_Alphabet_X)
    else:
        Alphabet_X, fill_value_Alphabet_X = _autocreate_alphabet(X,
                                                                 fill_value_X)
    if Alphabet_Y is not None:
        Alphabet_Y, fill_value_Alphabet_Y = _sanitise_array_input(Alphabet_Y,
                                                                  fill_value)
        Alphabet_Y, _ = _autocreate_alphabet(Alphabet_Y, fill_value_Alphabet_Y)
    else:
        Alphabet_Y, fill_value_Alphabet_Y = _autocreate_alphabet(Y,
                                                                 fill_value_Y)

    if X.size == 0:
        raise ValueError("arg X contains no elements")
    if Y.size == 0:
        raise ValueError("arg Y contains no elements")
    if np.any(_isnan(X)):
        raise ValueError("arg X contains NaN values")
    if np.any(_isnan(Y)):
        raise ValueError("arg Y contains NaN values")
    if Alphabet_X.size == 0:
        raise ValueError("arg Alphabet_X contains no elements")
    if np.any(_isnan(Alphabet_X)):
        raise ValueError("arg Alphabet_X contains NaN values")
    if Alphabet_Y.size == 0:
        raise ValueError("arg Alphabet_Y contains no elements")
    if np.any(_isnan(Alphabet_Y)):
        raise ValueError("arg Alphabet_Y contains NaN values")
    if _isnan(fill_value_X):
        raise ValueError("fill value for arg X is NaN")
    if _isnan(fill_value_Y):
        raise ValueError("fill value for arg Y is NaN")
    if X.shape[:-1] != Alphabet_X.shape[:-1]:
        raise ValueError("leading dimensions of args X and Alphabet_X do not "
                         "match")
    if Y.shape[:-1] != Alphabet_Y.shape[:-1]:
        raise ValueError("leading dimensions of args Y and Alphabet_Y do not "
                         "match")
    if not isinstance(norm_factor, str):
        raise ValueError("arg norm_factor not a string")
    if not cartesian_product and X.shape != Y.shape:
        raise ValueError("dimensions of args X and Y do not match")
    if cartesian_product and X.shape[-1] != Y.shape[-1]:
        raise ValueError("trailing dimensions of args X and Y do not match")
    # NB: No base parameter needed here, therefore no test!

    S, fill_value = _map_observations_to_integers((X, Alphabet_X,
                                                   Y, Alphabet_Y),
                                                  (fill_value_X,
                                                   fill_value_Alphabet_X,
                                                   fill_value_Y,
                                                   fill_value_Alphabet_Y))
    X, Alphabet_X, Y, Alphabet_Y = S

    I = information_mutual(X, Y, cartesian_product, fill_value=fill_value,
                           estimator=estimator, Alphabet_X=Alphabet_X,
                           Alphabet_Y=Alphabet_Y)

    norm_factor = norm_factor.upper().replace(' ', '')
    if norm_factor == 'Y':
        H2 = entropy(Y, fill_value=fill_value, estimator=estimator,
                     Alphabet_X=Alphabet_Y)
        H2 = np.reshape(H2, np.append(np.ones(I.ndim-H2.ndim),
                                      H2.shape).astype('int'))

        C = H2
    elif norm_factor == 'X':
        H1 = entropy(X, fill_value=fill_value, estimator=estimator,
                     Alphabet_X=Alphabet_X)
        H1 = np.reshape(H1, np.append(H1.shape,
                                      np.ones(I.ndim-H1.ndim)).astype('int'))

        C = H1
    elif norm_factor == 'Y+X' or norm_factor == 'X+Y':
        H1 = entropy(X, fill_value=fill_value, estimator=estimator,
                     Alphabet_X=Alphabet_X)
        H1 = np.reshape(H1, np.append(H1.shape,
                                      np.ones(I.ndim-H1.ndim)).astype('int'))
        H2 = entropy(Y, fill_value=fill_value, estimator=estimator,
                     Alphabet_X=Alphabet_Y)
        H2 = np.reshape(H2, np.append(np.ones(I.ndim-H2.ndim),
                                      H2.shape).astype('int'))

        C = H1 + H2
    elif norm_factor == 'MIN':
        H1 = entropy(X, fill_value=fill_value, estimator=estimator,
                     Alphabet_X=Alphabet_X)
        H1 = np.reshape(H1, np.append(H1.shape,
                                      np.ones(I.ndim-H1.ndim)).astype('int'))
        H2 = entropy(Y, fill_value=fill_value, estimator=estimator,
                     Alphabet_X=Alphabet_Y)
        H2 = np.reshape(H2, np.append(np.ones(I.ndim-H2.ndim),
                                      H2.shape).astype('int'))

        C = np.minimum(H1, H2)
    elif norm_factor == 'MAX':
        H1 = entropy(X, fill_value=fill_value, estimator=estimator,
                     Alphabet_X=Alphabet_X)
        H1 = np.reshape(H1, np.append(H1.shape,
                                      np.ones(I.ndim-H1.ndim)).astype('int'))
        H2 = entropy(Y, fill_value=fill_value, estimator=estimator,
                     Alphabet_X=Alphabet_Y)
        H2 = np.reshape(H2, np.append(np.ones(I.ndim-H2.ndim),
                                      H2.shape).astype('int'))

        C = np.maximum(H1, H2)
    elif norm_factor == 'XY' or norm_factor == 'YX':
        if not cartesian_product:
            H = np.empty(X.shape[:-1])
            if H.ndim > 0:
                H[:] = np.NaN
            else:
                H = np.float64(np.NaN)

            # Re-shape H and X, so that we may handle multi-dimensional arrays
            # equivalently and iterate across 0th axis
            # Re-shape X, so that we may handle multi-dimensional arrays
            # equivalently and iterate across 0th axis
            X = np.reshape(X, (-1, X.shape[-1]))
            # Re-shape Y, so that we may handle multi-dimensional arrays
            # equivalently and iterate across 0th axis
            Y = np.reshape(Y, (-1, Y.shape[-1]))
            Alphabet_X = np.reshape(Alphabet_X, (-1, Alphabet_X.shape[-1]))
            Alphabet_Y = np.reshape(Alphabet_Y, (-1, Alphabet_Y.shape[-1]))

            orig_shape_H = H.shape
            H = np.reshape(H, (-1, 1))

            for i in range(X.shape[0]):
                H[i] = entropy_joint(np.vstack((X[i], Y[i])),
                                     fill_value=fill_value,
                                     estimator=estimator,
                                     Alphabet_X=_vstack_pad((Alphabet_X[i],
                                                             Alphabet_Y[i]),
                                                            fill_value))
            H = np.reshape(H, orig_shape_H)

            C = H
        else:
            def f(X, Y, Alphabet_X, Alphabet_Y):
                return entropy_joint(np.vstack((X, Y)), fill_value=fill_value,
                                     estimator=estimator,
                                     Alphabet_X=_vstack_pad((Alphabet_X,
                                                             Alphabet_Y),
                                                            fill_value))
            H = _cartesian_product_apply(X, Y, f, Alphabet_X, Alphabet_Y)

            C = H
    elif norm_factor == 'SQRT':
        H1 = entropy(X, fill_value=fill_value, estimator=estimator,
                     Alphabet_X=Alphabet_X)
        H1 = np.reshape(H1, np.append(H1.shape,
                                      np.ones(I.ndim-H1.ndim)).astype('int'))
        H2 = entropy(Y, fill_value=fill_value, estimator=estimator,
                     Alphabet_X=Alphabet_Y)
        H2 = np.reshape(H2, np.append(np.ones(I.ndim-H2.ndim),
                                      H2.shape).astype('int'))

        C = np.sqrt(H1 * H2)
    else:
        raise ValueError("arg norm_factor has invalid value")

    I = I / C

    if keep_dims and not cartesian_product:
        I = I[..., np.newaxis]

    return I


def information_variation(X, Y=None, cartesian_product=False, base=2,
                          fill_value=-1, estimator='ML', Alphabet_X=None,
                          Alphabet_Y=None, keep_dims=False):
    """
    Returns the variation of information [Meil03] between arrays X and Y, each
    containing discrete random variable realisations.

    **Mathematical definition**:

    Given discrete random variables :math:`X`, :math:`Y`, the variation of
    information :math:`VI(X;Y)` is defined as:

    .. math::
        VI(X;Y) = H(X|Y) + H(Y|X)

    where :math:`H(\\cdot|\\cdot)` denotes the conditional entropy.

    **Estimation**:

    Variation of information is estimated based on frequency tables, using the
    following functions:

        entropy_joint()

        entropy()

    See below for a list of available estimators. Note that although variation
    of information is a non-negative quantity, depending on the chosen
    estimator the obtained estimate may be negative.

    **Parameters**:

    X,Y : numpy array (or array-like object such as a list of immutables, as \
    accepted by np.array())
        *cartesian_product==False and Y is not None*: X and Y are arrays
        containing discrete random variable realisations, with
        X.shape==Y.shape. Successive realisations of a random variable are
        indexed by the last axis in the respective arrays; multiple random
        variables in X and Y may be specified using preceding axes of the
        respective arrays (random variables are paired **one-to-one** between X
        and Y). When X.ndim==Y.ndim==1, returns a scalar. When X.ndim>1 and
        Y.ndim>1, returns an array of estimated information values with
        dimensions X.shape[:-1]. Neither X nor Y may contain (floating point)
        NaN values. Missing data may be specified using numpy masked arrays, as
        well as using standard numpy array/array-like objects; see below
        for details.

        *cartesian_product==True and Y is not None*: X and Y are arrays
        containing discrete random variable realisations, with
        X.shape[-1]==Y.shape[-1]. Successive realisations of a random variable
        are indexed by the last axis in the respective arrays; multiple random
        variables in X and Y may be specified using preceding axes of the
        respective arrays (random variables are paired **many-to-many** between
        X and Y). When X.ndim==Y.ndim==1, returns a scalar. When X.ndim>1 or
        Y.ndim>1, returns an array of estimated information values with
        dimensions np.append(X.shape[:-1],Y.shape[:-1]). Neither X nor Y may
        contain (floating point) NaN values. Missing data may be specified
        using numpy masked arrays, as well as using standard numpy
        array/array-like objects; see below for details.

        *Y is None*: Equivalent to information_variation(X, X, ... ). Thus, a
        shorthand syntax for computing variation of information (in bits)
        between all pairs of random variables in X is information_variation(X).
    cartesian_product : boolean
        Indicates whether random variables are paired **one-to-one** between X
        and Y (cartesian_product==False, the default value) or **many-to-many**
        between X and Y (cartesian_product==True).
    base : float
        The desired logarithmic base (default 2).
    fill_value : object
        It is possible to specify missing data using numpy masked arrays,
        pandas Series/DataFrames, as well as using standard numpy
        array/array-like objects with assigned placeholder values. When using
        numpy masked arrays, this function invokes np.ma.filled() internally,
        so that missing data are represented with the array's object-internal
        placeholder value fill_value (this function's fill_value parameter is
        ignored in such cases). When using pandas Series/DataFrames, an initial
        conversion to a numpy masked array is performed. When using standard
        numpy array/array-like objects, this function's fill_value parameter is
        used to specify the placeholder value for missing data (defaults to
        -1).

        Data equal to the placeholder value are subsequently ignored.
    estimator : str or float
        The desired estimator (see above for details on estimators). Possible
        values are:

            *'ML' (the default value)* : Maximum likelihood estimator.

            *any floating point value* : Maximum a posteriori esimator using
            Dirichlet prior (equivalent to maximum likelihood with pseudo-count
            for each outcome as specified).

            *PERKS* : Maximum a posteriori esimator using Dirichlet prior
            (equivalent to maximum likelihood with pseudo-count for each
            outcome set to 1/L, where L is the number of possible outcomes.

            *MINIMAX* : Maximum a posteriori esimator using Dirichlet prior
            (equivalent to maximum likelihood with pseudo-count for each
            outcome set to sqrt(N)/L, where N is the total number of
            realisations and where L is the number of possible outcomes.

            *JAMES-STEIN* : James-Stein estimator [HaSt09].

            *GOOD-TURING* : Good-Turing estimator [GaSa95].

    Alphabet_X, Alphabet_Y : numpy array (or array-like object such as a list \
    of immutables, as accepted by np.array())
        Respectively an array specifying the alphabet/alphabets of possible
        outcomes that random variable realisations in array X, Y may assume.
        Defaults to None, in which case the alphabet/alphabets of possible
        outcomes is/are implicitly based the observed outcomes in array X, Y
        respectively, with no additional, unobserved outcomes. In combination
        with any estimator other than maximum likelihood, it may be useful to
        specify alphabets including unobserved outcomes. For such cases,
        successive possible outcomes of a random variable are indexed by the
        last axis in Alphabet_X, Alphabet_Y respectively; multiple alphabets
        may be specified using preceding axes, with the requirement
        X.shape[:-1]==Alphabet_X.shape[:-1] (analogously for Y). Alphabets of
        different sizes may be specified either using numpy masked arrays, or
        by padding with the chosen placeholder fill_value.

        NB: When specifying alphabets, an alphabet of possible joint outcomes
        is always implicit from the alphabets of possible (marginal) outcomes
        in Alphabet_X, Alphabet_Y. For example, specifying
        Alphabet_X=Alphabet_Y=np.array(((1,2)) implies an alphabet of possible
        joint outcomes np.array(((1,1,2,2),(1,2,1,2))).
    keep_dims : boolean
        When set to True and cartesian_product==False an additional dimension
        of length one is appended to the returned array, facilitating any
        broadcast operations required by the user (defaults to False). Has no
        effect when cartesian_product==True.

    **Implementation notes**:

    Before estimation, outcomes are mapped to the set of non-negative integers
    internally, with the value -1 representing missing data. To avoid this
    internal conversion step, supply integer data and use the default fill
    value -1.
    """
    if Y is None:
        Y = X
        cartesian_product = True
        Alphabet_Y = Alphabet_X

    H1 = entropy_conditional(X, Y, cartesian_product, base, fill_value,
                             estimator, Alphabet_X, Alphabet_Y)
    H2 = entropy_conditional(Y, X, cartesian_product, base, fill_value,
                             estimator, Alphabet_Y, Alphabet_X)

    if cartesian_product:
        H2 = H2.T

    H = H1 + H2

    if keep_dims and not cartesian_product:
        H = H[..., np.newaxis]

    return H


def information_mutual(X, Y=None, cartesian_product=False, base=2,
                       fill_value=-1, estimator='ML', Alphabet_X=None,
                       Alphabet_Y=None, keep_dims=False):
    """
    Returns the mutual information (see e.g. [CoTh06]) between arrays X and Y,
    each containing discrete random variable realisations.

    **Mathematical definition**:

    Given discrete random variables :math:`X`, :math:`Y`, the mutual
    information :math:`I(X;Y)` is defined as:

    .. math::
        I(X;Y) = H(X) - H(X|Y)

    where :math:`H(\\cdot)` denotes the entropy and where
    :math:`H(\\cdot|\\cdot)` denotes the conditional entropy.

    **Estimation**:

    Mutual information is estimated based on frequency tables, using the
    following functions:

        entropy_joint()

        entropy()

    See below for a list of available estimators. Note that although mutual
    information is a non-negative quantity, depending on the chosen estimator
    the obtained estimate may be negative.

    **Parameters**:

    X,Y : numpy array (or array-like object such as a list of immutables, as \
    accepted by np.array())
        *cartesian_product==False and Y is not None*: X and Y are arrays
        containing discrete random variable realisations, with
        X.shape==Y.shape. Successive realisations of a random variable are
        indexed by the last axis in the respective arrays; multiple random
        variables in X and Y may be specified using preceding axes of the
        respective arrays (random variables are paired **one-to-one** between X
        and Y). When X.ndim==Y.ndim==1, returns a scalar. When X.ndim>1 and
        Y.ndim>1, returns an array of estimated mutual information values with
        dimensions X.shape[:-1]. Neither X nor Y may contain (floating point)
        NaN values. Missing data may be specified using numpy masked arrays, as
        well as using standard numpy array/array-like objects; see below
        for details.

        *cartesian_product==True and Y is not None*: X and Y are arrays
        containing discrete random variable realisations, with
        X.shape[-1]==Y.shape[-1]. Successive realisations of a random variable
        are indexed by the last axis in the respective arrays; multiple random
        variables in X and Y may be specified using preceding axes of the
        respective arrays (random variables are paired **many-to-many** between
        X and Y). When X.ndim==Y.ndim==1, returns a scalar. When X.ndim>1 or
        Y.ndim>1, returns an array of estimated mutual information values with
        dimensions np.append(X.shape[:-1],Y.shape[:-1]). Neither X nor Y may
        contain (floating point) NaN values. Missing data may be specified
        using numpy masked arrays, as well as using standard numpy
        array/array-like objects; see below for details.

        *Y is None*: Equivalent to information_mutual(X, X, ... ). Thus, a
        shorthand syntax for computing mutual information (in bits) between all
        pairs of random variables in X is information_mutual(X).
    cartesian_product : boolean
        Indicates whether random variables are paired **one-to-one** between X
        and Y (cartesian_product==False, the default value) or **many-to-many**
        between X and Y (cartesian_product==True).
    base : float
        The desired logarithmic base (default 2).
    fill_value : object
        It is possible to specify missing data using numpy masked arrays,
        pandas Series/DataFrames, as well as using standard numpy
        array/array-like objects with assigned placeholder values. When using
        numpy masked arrays, this function invokes np.ma.filled() internally,
        so that missing data are represented with the array's object-internal
        placeholder value fill_value (this function's fill_value parameter is
        ignored in such cases). When using pandas Series/DataFrames, an initial
        conversion to a numpy masked array is performed. When using standard
        numpy array/array-like objects, this function's fill_value parameter is
        used to specify the placeholder value for missing data (defaults to
        -1).

        Data equal to the placeholder value are subsequently ignored.
    estimator : str or float
        The desired estimator (see above for details on estimators). Possible
        values are:

            *'ML' (the default value)* : Maximum likelihood estimator.

            *any floating point value* : Maximum a posteriori esimator using
            Dirichlet prior (equivalent to maximum likelihood with pseudo-count
            for each outcome as specified).

            *PERKS* : Maximum a posteriori esimator using Dirichlet prior
            (equivalent to maximum likelihood with pseudo-count for each
            outcome set to 1/L, where L is the number of possible outcomes.

            *MINIMAX* : Maximum a posteriori esimator using Dirichlet prior
            (equivalent to maximum likelihood with pseudo-count for each
            outcome set to sqrt(N)/L, where N is the total number of
            realisations and where L is the number of possible outcomes.

            *JAMES-STEIN* : James-Stein estimator [HaSt09].

            *GOOD-TURING* : Good-Turing estimator [GaSa95].

    Alphabet_X, Alphabet_Y : numpy array (or array-like object such as a list \
    of immutables, as accepted by np.array())
        Respectively an array specifying the alphabet/alphabets of possible
        outcomes that random variable realisations in array X, Y may assume.
        Defaults to None, in which case the alphabet/alphabets of possible
        outcomes is/are implicitly based the observed outcomes in array X, Y
        respectively, with no additional, unobserved outcomes. In combination
        with any estimator other than maximum likelihood, it may be useful to
        specify alphabets including unobserved outcomes. For such cases,
        successive possible outcomes of a random variable are indexed by the
        last axis in Alphabet_X, Alphabet_Y respectively; multiple alphabets
        may be specified using preceding axes, with the requirement
        X.shape[:-1]==Alphabet_X.shape[:-1] (analogously for Y). Alphabets of
        different sizes may be specified either using numpy masked arrays, or
        by padding with the chosen placeholder fill_value.

        NB: When specifying alphabets, an alphabet of possible joint outcomes
        is always implicit from the alphabets of possible (marginal) outcomes
        in Alphabet_X, Alphabet_Y. For example, specifying
        Alphabet_X=Alphabet_Y=np.array(((1,2)) implies an alphabet of possible
        joint outcomes np.array(((1,1,2,2),(1,2,1,2))).
    keep_dims : boolean
        When set to True and cartesian_product==False an additional dimension
        of length one is appended to the returned array, facilitating any
        broadcast operations required by the user (defaults to False). Has no
        effect when cartesian_product==True.

    **Implementation notes**:

    Before estimation, outcomes are mapped to the set of non-negative integers
    internally, with the value -1 representing missing data. To avoid this
    internal conversion step, supply integer data and use the default fill
    value -1.
    """
    H_conditional = entropy_conditional(X, Y, cartesian_product, base,
                                        fill_value, estimator, Alphabet_X,
                                        Alphabet_Y)
    H = entropy(X, base, fill_value, estimator, Alphabet_X)

    H = np.reshape(H, np.append(H.shape,
                                np.ones(H_conditional.ndim -
                                        H.ndim)).astype('int'))

    H = H - H_conditional

    if keep_dims and not cartesian_product:
        H = H[..., np.newaxis]

    return H


def entropy_cross(X, Y=None, cartesian_product=False, base=2, fill_value=-1,
                  estimator='ML', Alphabet_X=None, Alphabet_Y=None,
                  keep_dims=False):
    """
    Returns the cross entropy (see e.g. [Murp12]) between arrays X and Y, each
    containing discrete random variable realisations.

    **Mathematical definition**:

    Denoting with :math:`P_X(x)`, :math:`P_Y(x)` respectively the probability
    of observing an outcome :math:`x` with discrete random variables :math:`X`,
    :math:`Y`, the cross entropy :math:`H^\\times(X,Y)` is defined as:

    .. math::
        H^\\times(X,Y) = -\\sum_x {P_X(x) \\log {P_Y(x)}}.

    **Estimation**:

    Cross entropy is estimated based on frequency tables. See below for a list
    of available estimators.

    **Parameters**:

    X,Y : numpy array (or array-like object such as a list of immutables, as \
    accepted by np.array())
        *cartesian_product==False and Y is not None*: X and Y are arrays
        containing discrete random variable realisations, with
        X.shape[:-1]==Y.shape[:-1]. Successive realisations of a random
        variable are indexed by the last axis in the respective arrays;
        multiple random variables in X and Y may be specified using preceding
        axes of the respective arrays (random variables are paired
        **one-to-one** between X and Y). When X.ndim==Y.ndim==1, returns a
        scalar. When X.ndim>1 and Y.ndim>1, returns an array of estimated cross
        entropies with dimensions X.shape[:-1]. Neither X nor Y may contain
        (floating point) NaN values. Missing data may be specified using numpy
        masked arrays, as well as using standard numpy array/array-like
        objects; see below for details.

        *cartesian_product==True and Y is not None*: X and Y are arrays
        containing discrete random variable realisations. Successive
        realisations of a random variable are indexed by the last axis in the
        respective arrays; multiple random variables in X and Y may be
        specified using preceding axes of the respective arrays (random
        variables are paired **many-to-many** between X and Y). When
        X.ndim==Y.ndim==1, returns a scalar. When X.ndim>1 or Y.ndim>1, returns
        an array of estimated cross entropies with dimensions
        np.append(X.shape[:-1],Y.shape[:-1]). Neither X nor Y may contain
        (floating point) NaN values. Missing data may be specified using numpy
        masked arrays, as well as using standard numpy array/array-like
        objects; see below for details.

        *Y is None*: Equivalent to entropy_cross(X, X, ... ). Thus, a shorthand
        syntax for computing cross entropies (in bits) between all pairs of
        random variables in X is entropy_cross(X).
    cartesian_product : boolean
        Indicates whether random variables are paired **one-to-one** between X
        and Y (cartesian_product==False, the default value) or **many-to-many**
        between X and Y (cartesian_product==True).
    base : float
        The desired logarithmic base (default 2).
    fill_value : object
        It is possible to specify missing data using numpy masked arrays,
        pandas Series/DataFrames, as well as using standard numpy
        array/array-like objects with assigned placeholder values. When using
        numpy masked arrays, this function invokes np.ma.filled() internally,
        so that missing data are represented with the array's object-internal
        placeholder value fill_value (this function's fill_value parameter is
        ignored in such cases). When using pandas Series/DataFrames, an initial
        conversion to a numpy masked array is performed. When using standard
        numpy array/array-like objects, this function's fill_value parameter is
        used to specify the placeholder value for missing data (defaults to
        -1).

        Data equal to the placeholder value are subsequently ignored.
    estimator : str or float
        The desired estimator (see above for details on estimators). Possible
        values are:

            *'ML' (the default value)* : Maximum likelihood estimator.

            *any floating point value* : Maximum a posteriori esimator using
            Dirichlet prior (equivalent to maximum likelihood with pseudo-count
            for each outcome as specified).

            *PERKS* : Maximum a posteriori esimator using Dirichlet prior
            (equivalent to maximum likelihood with pseudo-count for each
            outcome set to 1/L, where L is the number of possible outcomes.

            *MINIMAX* : Maximum a posteriori esimator using Dirichlet prior
            (equivalent to maximum likelihood with pseudo-count for each
            outcome set to sqrt(N)/L, where N is the total number of
            realisations and where L is the number of possible outcomes.

            *JAMES-STEIN* : James-Stein estimator [HaSt09].

            *GOOD-TURING* : Good-Turing estimator [GaSa95].

    Alphabet_X, Alphabet_Y : numpy array (or array-like object such as a list \
    of immutables, as accepted by np.array())
        Respectively an array specifying the alphabet/alphabets of possible
        outcomes that random variable realisations in array X, Y may assume.
        Defaults to None, in which case the alphabet/alphabets of possible
        outcomes is/are implicitly based the observed outcomes in array X, Y
        respectively, with no additional, unobserved outcomes. In combination
        with any estimator other than maximum likelihood, it may be useful to
        specify alphabets including unobserved outcomes. For such cases,
        successive possible outcomes of a random variable are indexed by the
        last axis in Alphabet_X, Alphabet_Y respectively; multiple alphabets
        may be specified using preceding axes, with the requirement
        X.shape[:-1]==Alphabet_X.shape[:-1] (analogously for Y). Alphabets of
        different sizes may be specified either using numpy masked arrays, or
        by padding with the chosen placeholder fill_value.
    keep_dims : boolean
        When set to True and cartesian_product==False an additional dimension
        of length one is appended to the returned array, facilitating any
        broadcast operations required by the user (defaults to False). Has no
        effect when cartesian_product==True.

    **Implementation notes**:

    Before estimation, outcomes are mapped to the set of non-negative integers
    internally, with the value -1 representing missing data. To avoid this
    internal conversion step, supply integer data and use the default fill
    value -1.

    """
    if Y is None:
        Y = X
        cartesian_product = True
        Alphabet_Y = Alphabet_X

    X, fill_value_X = _sanitise_array_input(X, fill_value)
    Y, fill_value_Y = _sanitise_array_input(Y, fill_value)
    if Alphabet_X is not None:
        Alphabet_X, fill_value_Alphabet_X = _sanitise_array_input(Alphabet_X,
                                                                  fill_value)
        Alphabet_X, _ = _autocreate_alphabet(Alphabet_X, fill_value_Alphabet_X)
    else:
        Alphabet_X, fill_value_Alphabet_X = _autocreate_alphabet(X,
                                                                 fill_value_X)
    if Alphabet_Y is not None:
        Alphabet_Y, fill_value_Alphabet_Y = _sanitise_array_input(Alphabet_Y,
                                                                  fill_value)
        Alphabet_Y, _ = _autocreate_alphabet(Alphabet_Y, fill_value_Alphabet_Y)
    else:
        Alphabet_Y, fill_value_Alphabet_Y = _autocreate_alphabet(Y,
                                                                 fill_value_Y)

    if X.size == 0:
        raise ValueError("arg X contains no elements")
    if Y.size == 0:
        raise ValueError("arg Y contains no elements")
    if np.any(_isnan(X)):
        raise ValueError("arg X contains NaN values")
    if np.any(_isnan(Y)):
        raise ValueError("arg Y contains NaN values")
    if Alphabet_X.size == 0:
        raise ValueError("arg Alphabet_X contains no elements")
    if np.any(_isnan(Alphabet_X)):
        raise ValueError("arg Alphabet_X contains NaN values")
    if Alphabet_Y.size == 0:
        raise ValueError("arg Alphabet_Y contains no elements")
    if np.any(_isnan(Alphabet_Y)):
        raise ValueError("arg Alphabet_Y contains NaN values")
    if _isnan(fill_value_X):
        raise ValueError("fill value for arg X is NaN")
    if _isnan(fill_value_Y):
        raise ValueError("fill value for arg Y is NaN")
    if X.shape[:-1] != Alphabet_X.shape[:-1]:
        raise ValueError("leading dimensions of args X and Alphabet_X do not "
                         "match")
    if Y.shape[:-1] != Alphabet_Y.shape[:-1]:
        raise ValueError("leading dimensions of args Y and Alphabet_Y do not "
                         "match")
    if not cartesian_product and X.shape[:-1] != Y.shape[:-1]:
        raise ValueError("dimensions of args X and Y do not match")
    if not (np.isscalar(base) and np.isreal(base) and base > 0):
        raise ValueError("arg base not a positive real-valued scalar")

    S, fill_value = _map_observations_to_integers((X, Alphabet_X,
                                                   Y, Alphabet_Y),
                                                  (fill_value_X,
                                                   fill_value_Alphabet_X,
                                                   fill_value_Y,
                                                   fill_value_Alphabet_Y))
    X, Alphabet_X, Y, Alphabet_Y = S

    if not cartesian_product:
        H = np.empty(X.shape[:-1])
        if H.ndim > 0:
            H[:] = np.NaN
        else:
            H = np.float64(np.NaN)
    else:
        def f(X, Y, Alphabet_X, Alphabet_Y):
            return entropy_cross(X, Y, False, base, fill_value, estimator,
                                 Alphabet_X, Alphabet_Y)
        return _cartesian_product_apply(X, Y, f, Alphabet_X, Alphabet_Y)

    # Re-shape H, X and Y, so that we may handle multi-dimensional arrays
    # equivalently and iterate across 0th axis
    X = np.reshape(X, (-1, X.shape[-1]))
    Y = np.reshape(Y, (-1, Y.shape[-1]))
    Alphabet_X = np.reshape(Alphabet_X, (-1, Alphabet_X.shape[-1]))
    Alphabet_Y = np.reshape(Alphabet_Y, (-1, Alphabet_Y.shape[-1]))
    orig_shape_H = H.shape
    H = np.reshape(H, (-1, 1))

    _verify_alphabet_sufficiently_large(X, Alphabet_X, fill_value)
    _verify_alphabet_sufficiently_large(Y, Alphabet_Y, fill_value)

    # NB: Observations are not considered jointly, thus elements in each row
    # are sorted independently
    X = np.sort(X, axis=1)
    Y = np.sort(Y, axis=1)

    # Compute symbol run-lengths
    # Compute symbol change indicators
    B = X[:, 1:] != X[:, :-1]
    C = Y[:, 1:] != Y[:, :-1]
    for i in range(X.shape[0]):
        # Obtain symbol change positions
        I = np.append(np.where(B[i]), X.shape[1]-1)
        # Compute run lengths
        L = np.diff(np.append(-1, I))

        alphabet_X = X[i, I]
        if estimator != 'ML':
            L, alphabet_X = _append_empty_bins_using_alphabet(L, alphabet_X,
                                                              Alphabet_X[i],
                                                              fill_value)
        L, alphabet_X = _remove_counts_at_fill_value(L, alphabet_X, fill_value)
        if not np.any(L):
            continue
        P1, _ = _estimate_probabilities(L, estimator)

        # Obtain symbol change positions
        J = np.append(np.where(C[i]), Y.shape[1]-1)
        # Compute run lengths
        L = np.diff(np.append(-1, J))

        alphabet_Y = Y[i, J]
        if estimator != 'ML':
            L, alphabet_Y = _append_empty_bins_using_alphabet(L, alphabet_Y,
                                                              Alphabet_Y[i],
                                                              fill_value)
        L, alphabet_Y = _remove_counts_at_fill_value(L, alphabet_Y, fill_value)
        if not np.any(L):
            continue
        P2, _ = _estimate_probabilities(L, estimator)

        # Merge probability distributions, so that common symbols have common
        # array location
        Alphabet = np.union1d(alphabet_X, alphabet_Y)
        P = np.zeros_like(Alphabet, dtype=P1.dtype)
        Q = np.zeros_like(Alphabet, dtype=P2.dtype)
        P[np.in1d(Alphabet, alphabet_X, assume_unique=True)] = P1
        Q[np.in1d(Alphabet, alphabet_Y, assume_unique=True)] = P2

        H[i] = entropy_cross_pmf(P, Q, False, base)

    # Reverse re-shaping
    H = np.reshape(H, orig_shape_H)

    if keep_dims and not cartesian_product:
        H = H[..., np.newaxis]

    return H


def divergence_kullbackleibler(X, Y=None, cartesian_product=False, base=2,
                               fill_value=-1, estimator='ML', Alphabet_X=None,
                               Alphabet_Y=None, keep_dims=False):
    """
    Returns the Kullback-Leibler divergence (see e.g. [CoTh06]) between arrays
    X and Y, each containing discrete random variable realisations.

    **Mathematical definition**:

    Denoting with :math:`P_X(x)`, :math:`P_Y(x)` respectively the probability
    of observing an outcome :math:`x` with discrete random variables :math:`X`,
    :math:`Y`, the Kullback-Leibler divergence
    :math:`D_{\\mathrm{KL}}(P_X\\parallel P_Y)` is defined as:

    .. math::
        D_{\\mathrm{KL}}(P_X \\parallel P_Y) =
        -\\sum_x {P_X(x) \\log {\\frac{P_Y(x)}{P_X(x)}}}.

    **Estimation**:

    Kullback-Leibler divergence is estimated based on frequency tables, using
    the following functions:

        entropy_cross()

        entropy()

    See below for a list of available estimators. Note that although
    Kullback-Leibler divergence is a non-negative quantity, depending on the
    chosen estimator the obtained estimate may be negative.

    **Parameters**:

    X,Y : numpy array (or array-like object such as a list of immutables, as \
    accepted by np.array())
        *cartesian_product==False and Y is not None*: X and Y are arrays
        containing discrete random variable realisations, with
        X.shape[:-1]==Y.shape[:-1]. Successive realisations of a random
        variable are indexed by the last axis in the respective arrays;
        multiple random variables in X and Y may be specified using preceding
        axes of the respective arrays (random variables are paired
        **one-to-one** between X and Y). When X.ndim==Y.ndim==1, returns a
        scalar. When X.ndim>1 and Y.ndim>1, returns an array of estimated
        divergence values with dimensions X.shape[:-1]. Neither X nor Y may
        contain (floating point) NaN values. Missing data may be specified
        using numpy masked arrays, as well as using standard numpy
        array/array-like objects; see below for details.

        *cartesian_product==True and Y is not None*: X and Y are arrays
        containing discrete random variable realisations. Successive
        realisations of a random variable are indexed by the last axis in the
        respective arrays; multiple random variables in X and Y may be
        specified using preceding axes of the respective arrays (random
        variables are paired **many-to-many** between X and Y). When
        X.ndim==Y.ndim==1, returns a scalar. When X.ndim>1 or Y.ndim>1, returns
        an array of estimated divergence values with dimensions
        np.append(X.shape[:-1],Y.shape[:-1]). Neither X nor Y may contain
        (floating point) NaN values. Missing data may be specified using numpy
        masked arrays, as well as using standard numpy array/array-like
        objects; see below for details.

        *Y is None*: Equivalent to divergence_kullbackleibler(X, X, ... ).
        Thus, a shorthand syntax for computing Kullback-Leibler divergence (in
        bits) between all pairs of random variables in X is
        divergence_kullbackleibler(X).
    cartesian_product : boolean
        Indicates whether random variables are paired **one-to-one** between X
        and Y (cartesian_product==False, the default value) or **many-to-many**
        between X and Y (cartesian_product==True).
    base : float
        The desired logarithmic base (default 2).
    fill_value : object
        It is possible to specify missing data using numpy masked arrays,
        pandas Series/DataFrames, as well as using standard numpy
        array/array-like objects with assigned placeholder values. When using
        numpy masked arrays, this function invokes np.ma.filled() internally,
        so that missing data are represented with the array's object-internal
        placeholder value fill_value (this function's fill_value parameter is
        ignored in such cases). When using pandas Series/DataFrames, an initial
        conversion to a numpy masked array is performed. When using standard
        numpy array/array-like objects, this function's fill_value parameter is
        used to specify the placeholder value for missing data (defaults to
        -1).

        Data equal to the placeholder value are subsequently ignored.
    estimator : str or float
        The desired estimator (see above for details on estimators). Possible
        values are:

            *'ML' (the default value)* : Maximum likelihood estimator.

            *any floating point value* : Maximum a posteriori esimator using
            Dirichlet prior (equivalent to maximum likelihood with pseudo-count
            for each outcome as specified).

            *PERKS* : Maximum a posteriori esimator using Dirichlet prior
            (equivalent to maximum likelihood with pseudo-count for each
            outcome set to 1/L, where L is the number of possible outcomes.

            *MINIMAX* : Maximum a posteriori esimator using Dirichlet prior
            (equivalent to maximum likelihood with pseudo-count for each
            outcome set to sqrt(N)/L, where N is the total number of
            realisations and where L is the number of possible outcomes.

            *JAMES-STEIN* : James-Stein estimator [HaSt09].

            *GOOD-TURING* : Good-Turing estimator [GaSa95].

    Alphabet_X, Alphabet_Y : numpy array (or array-like object such as a list \
    of immutables, as accepted by np.array())
        Respectively an array specifying the alphabet/alphabets of possible
        outcomes that random variable realisations in array X, Y may assume.
        Defaults to None, in which case the alphabet/alphabets of possible
        outcomes is/are implicitly based the observed outcomes in array X, Y
        respectively, with no additional, unobserved outcomes. In combination
        with any estimator other than maximum likelihood, it may be useful to
        specify alphabets including unobserved outcomes. For such cases,
        successive possible outcomes of a random variable are indexed by the
        last axis in Alphabet_X, Alphabet_Y respectively; multiple alphabets
        may be specified using preceding axes, with the requirement
        X.shape[:-1]==Alphabet_X.shape[:-1] (analogously for Y). Alphabets of
        different sizes may be specified either using numpy masked arrays, or
        by padding with the chosen placeholder fill_value.
    keep_dims : boolean
        When set to True and cartesian_product==False an additional dimension
        of length one is appended to the returned array, facilitating any
        broadcast operations required by the user (defaults to False). Has no
        effect when cartesian_product==True.

    **Implementation notes**:

    Before estimation, outcomes are mapped to the set of non-negative integers
    internally, with the value -1 representing missing data. To avoid this
    internal conversion step, supply integer data and use the default fill
    value -1.

    """
    H_cross = entropy_cross(X, Y, cartesian_product, base, fill_value,
                            estimator, Alphabet_X, Alphabet_Y)
    H = entropy(X, base, fill_value, estimator, Alphabet_X)

    H = np.reshape(H, np.append(H.shape,
                                np.ones(H_cross.ndim-H.ndim)).astype('int'))

    H = H_cross - H

    if keep_dims and not cartesian_product:
        H = H[..., np.newaxis]

    return H


def divergence_jensenshannon(X, Y=None, cartesian_product=False, base=2,
                             fill_value=-1, estimator='ML', Alphabet_X=None,
                             Alphabet_Y=None, keep_dims=False):
    """
    Returns the Jensen-Shannon divergence [Lin91] between arrays X and Y, each
    containing discrete random variable realisations.

    **Mathematical definition**:

    Denoting with :math:`P_X`, :math:`P_Y` respectively probability
    distributions with common domain, associated with discrete random variables
    :math:`X`, :math:`Y`, the Jensen-Shannon divergence
    :math:`D_{\\mathrm{JS}}(P_X \\parallel P_Y)` is defined as:

    .. math::
        D_{\\mathrm{JS}}(P_X \\parallel P_Y) =
        \\frac{1}{2} D_{\\mathrm{KL}}(P_X \\parallel M) +
        \\frac{1}{2} D_{\\mathrm{KL}}(P_Y \\parallel M)

    where :math:`M = \\frac{1}{2}(P_X + P_Y)` and where
    :math:`D_{\\mathrm{KL}}(\\cdot \\parallel \\cdot)` denotes the
    Kullback-Leibler divergence.

    **Estimation**:

    Jensen-Shannon divergence is estimated based on frequency tables. See below
    for a list of available estimators.

    **Parameters**:

    X,Y : numpy array (or array-like object such as a list of immutables, as \
    accepted by np.array())
        *cartesian_product==False and Y is not None*: X and Y are arrays
        containing discrete random variable realisations, with
        X.shape[:-1]==Y.shape[:-1]. Successive realisations of a random
        variable are indexed by the last axis in the respective arrays;
        multiple random variables in X and Y may be specified using preceding
        axes of the respective arrays (random variables are paired
        **one-to-one** between X and Y). When X.ndim==Y.ndim==1, returns a
        scalar. When X.ndim>1 and Y.ndim>1, returns an array of estimated
        divergence values with dimensions X.shape[:-1]. Neither X nor Y may
        contain (floating point) NaN values. Missing data may be specified
        using numpy masked arrays, as well as using standard numpy
        array/array-like objects; see below for details.

        *cartesian_product==True and Y is not None*: X and Y are arrays
        containing discrete random variable realisations. Successive
        realisations of a random variable are indexed by the last axis in the
        respective arrays; multiple random variables in X and Y may be
        specified using preceding axes of the respective arrays (random
        variables are paired **many-to-many** between X and Y). When
        X.ndim==Y.ndim==1, returns a scalar. When X.ndim>1 or Y.ndim>1, returns
        an array of estimated divergence values with dimensions
        np.append(X.shape[:-1],Y.shape[:-1]). Neither X nor Y may contain
        (floating point) NaN values. Missing data may be specified using numpy
        masked arrays, as well as using standard numpy array/array-like
        objects; see below for details.

        *Y is None*: Equivalent to divergence_jensenshannon(X, X, ... ). Thus,
        a shorthand syntax for computing Jensen-Shannon divergence (in bits)
        between all pairs of random variables in X is
        divergence_jensenshannon(X).
    cartesian_product : boolean
        Indicates whether random variables are paired **one-to-one** between X
        and Y (cartesian_product==False, the default value) or **many-to-many**
        between X and Y (cartesian_product==True).
    base : float
        The desired logarithmic base (default 2).
    fill_value : object
        It is possible to specify missing data using numpy masked arrays,
        pandas Series/DataFrames, as well as using standard numpy
        array/array-like objects with assigned placeholder values. When using
        numpy masked arrays, this function invokes np.ma.filled() internally,
        so that missing data are represented with the array's object-internal
        placeholder value fill_value (this function's fill_value parameter is
        ignored in such cases). When using pandas Series/DataFrames, an initial
        conversion to a numpy masked array is performed. When using standard
        numpy array/array-like objects, this function's fill_value parameter is
        used to specify the placeholder value for missing data (defaults to
        -1).

        Data equal to the placeholder value are subsequently ignored.
    estimator : str or float
        The desired estimator (see above for details on estimators). Possible
        values are:

            *'ML' (the default value)* : Maximum likelihood estimator.

            *any floating point value* : Maximum a posteriori esimator using
            Dirichlet prior (equivalent to maximum likelihood with pseudo-count
            for each outcome as specified).

            *PERKS* : Maximum a posteriori esimator using Dirichlet prior
            (equivalent to maximum likelihood with pseudo-count for each
            outcome set to 1/L, where L is the number of possible outcomes.

            *MINIMAX* : Maximum a posteriori esimator using Dirichlet prior
            (equivalent to maximum likelihood with pseudo-count for each
            outcome set to sqrt(N)/L, where N is the total number of
            realisations and where L is the number of possible outcomes.

            *JAMES-STEIN* : James-Stein estimator [HaSt09].

            *GOOD-TURING* : Good-Turing estimator [GaSa95].

    Alphabet_X, Alphabet_Y : numpy array (or array-like object such as a list \
    of immutables, as accepted by np.array())
        Respectively an array specifying the alphabet/alphabets of possible
        outcomes that random variable realisations in array X, Y may assume.
        Defaults to None, in which case the alphabet/alphabets of possible
        outcomes is/are implicitly based the observed outcomes in array X, Y
        respectively, with no additional, unobserved outcomes. In combination
        with any estimator other than maximum likelihood, it may be useful to
        specify alphabets including unobserved outcomes. For such cases,
        successive possible outcomes of a random variable are indexed by the
        last axis in Alphabet_X, Alphabet_Y respectively; multiple alphabets
        may be specified using preceding axes, with the requirement
        X.shape[:-1]==Alphabet_X.shape[:-1] (analogously for Y). Alphabets of
        different sizes may be specified either using numpy masked arrays, or
        by padding with the chosen placeholder fill_value.
    keep_dims : boolean
        When set to True and cartesian_product==False an additional dimension
        of length one is appended to the returned array, facilitating any
        broadcast operations required by the user (defaults to False). Has no
        effect when cartesian_product==True.

    **Implementation notes**:

    Before estimation, outcomes are mapped to the set of non-negative integers
    internally, with the value -1 representing missing data. To avoid this
    internal conversion step, supply integer data and use the default fill
    value -1.

    """
    if Y is None:
        Y = X
        cartesian_product = True
        Alphabet_Y = Alphabet_X

    X, fill_value_X = _sanitise_array_input(X, fill_value)
    Y, fill_value_Y = _sanitise_array_input(Y, fill_value)
    if Alphabet_X is not None:
        Alphabet_X, fill_value_Alphabet_X = _sanitise_array_input(Alphabet_X,
                                                                  fill_value)
        Alphabet_X, _ = _autocreate_alphabet(Alphabet_X, fill_value_Alphabet_X)
    else:
        Alphabet_X, fill_value_Alphabet_X = _autocreate_alphabet(X,
                                                                 fill_value_X)
    if Alphabet_Y is not None:
        Alphabet_Y, fill_value_Alphabet_Y = _sanitise_array_input(Alphabet_Y,
                                                                  fill_value)
        Alphabet_Y, _ = _autocreate_alphabet(Alphabet_Y, fill_value_Alphabet_Y)
    else:
        Alphabet_Y, fill_value_Alphabet_Y = _autocreate_alphabet(Y,
                                                                 fill_value_Y)

    if X.size == 0:
        raise ValueError("arg X contains no elements")
    if Y.size == 0:
        raise ValueError("arg Y contains no elements")
    if np.any(_isnan(X)):
        raise ValueError("arg X contains NaN values")
    if np.any(_isnan(Y)):
        raise ValueError("arg Y contains NaN values")
    if Alphabet_X.size == 0:
        raise ValueError("arg Alphabet_X contains no elements")
    if np.any(_isnan(Alphabet_X)):
        raise ValueError("arg Alphabet_X contains NaN values")
    if Alphabet_Y.size == 0:
        raise ValueError("arg Alphabet_Y contains no elements")
    if np.any(_isnan(Alphabet_Y)):
        raise ValueError("arg Alphabet_Y contains NaN values")
    if _isnan(fill_value_X):
        raise ValueError("fill value for arg X is NaN")
    if _isnan(fill_value_Y):
        raise ValueError("fill value for arg Y is NaN")
    if X.shape[:-1] != Alphabet_X.shape[:-1]:
        raise ValueError("leading dimensions of args X and Alphabet_X do not "
                         "match")
    if Y.shape[:-1] != Alphabet_Y.shape[:-1]:
        raise ValueError("leading dimensions of args Y and Alphabet_Y do not "
                         "match")
    if not cartesian_product and X.shape[:-1] != Y.shape[:-1]:
        raise ValueError("dimensions of args X and Y do not match")
    if not (np.isscalar(base) and np.isreal(base) and base > 0):
        raise ValueError("arg base not a positive real-valued scalar")

    S, fill_value = _map_observations_to_integers((X, Alphabet_X,
                                                   Y, Alphabet_Y),
                                                  (fill_value_X,
                                                   fill_value_Alphabet_X,
                                                   fill_value_Y,
                                                   fill_value_Alphabet_Y))
    X, Alphabet_X, Y, Alphabet_Y = S

    if not cartesian_product:
        H = np.empty(X.shape[:-1])
        if H.ndim > 0:
            H[:] = np.NaN
        else:
            H = np.float64(np.NaN)
    else:
        def f(X, Y, Alphabet_X, Alphabet_Y):
            return divergence_jensenshannon(X, Y, False, base, fill_value,
                                            estimator, Alphabet_X, Alphabet_Y)
        return _cartesian_product_apply(X, Y, f, Alphabet_X, Alphabet_Y)

    # Re-shape H, X and Y, so that we may handle multi-dimensional arrays
    # equivalently and iterate across 0th axis
    X = np.reshape(X, (-1, X.shape[-1]))
    Y = np.reshape(Y, (-1, Y.shape[-1]))
    Alphabet_X = np.reshape(Alphabet_X, (-1, Alphabet_X.shape[-1]))
    Alphabet_Y = np.reshape(Alphabet_Y, (-1, Alphabet_Y.shape[-1]))
    orig_shape_H = H.shape
    H = np.reshape(H, (-1, 1))

    _verify_alphabet_sufficiently_large(X, Alphabet_X, fill_value)
    _verify_alphabet_sufficiently_large(Y, Alphabet_Y, fill_value)

    # NB: Observations are not considered jointly, thus elements in each row
    # are sorted independently
    X = np.sort(X, axis=1)
    Y = np.sort(Y, axis=1)

    # Compute symbol run-lengths
    # Compute symbol change indicators
    B = X[:, 1:] != X[:, :-1]
    C = Y[:, 1:] != Y[:, :-1]
    for i in range(X.shape[0]):
        # Obtain symbol change positions
        I = np.append(np.where(B[i]), X.shape[1]-1)
        # Compute run lengths
        L = np.diff(np.append(-1, I))

        alphabet_X = X[i, I]
        if estimator != 'ML':
            L, alphabet_X = _append_empty_bins_using_alphabet(L, alphabet_X,
                                                              Alphabet_X[i],
                                                              fill_value)
        L, alphabet_X = _remove_counts_at_fill_value(L, alphabet_X, fill_value)
        if not np.any(L):
            continue
        P1, _ = _estimate_probabilities(L, estimator)

        # Obtain symbol change positions
        J = np.append(np.where(C[i]), Y.shape[1]-1)
        # Compute run lengths
        L = np.diff(np.append(-1, J))

        alphabet_Y = Y[i, J]
        if estimator != 'ML':
            L, alphabet_Y = _append_empty_bins_using_alphabet(L, alphabet_Y,
                                                              Alphabet_Y[i],
                                                              fill_value)
        L, alphabet_Y = _remove_counts_at_fill_value(L, alphabet_Y, fill_value)
        if not np.any(L):
            continue
        P2, _ = _estimate_probabilities(L, estimator)

        # Merge probability distributions, so that common symbols have common
        # array location
        Alphabet = np.union1d(alphabet_X, alphabet_Y)
        P = np.zeros_like(Alphabet, dtype=P1.dtype)
        Q = np.zeros_like(Alphabet, dtype=P2.dtype)
        P[np.in1d(Alphabet, alphabet_X, assume_unique=True)] = P1
        Q[np.in1d(Alphabet, alphabet_Y, assume_unique=True)] = P2

        H[i] = entropy_pmf(0.5*P + 0.5*Q, base) - \
            0.5*entropy_pmf(P, base) - 0.5*entropy_pmf(Q, base)

    # Reverse re-shaping
    H = np.reshape(H, orig_shape_H)

    if keep_dims and not cartesian_product:
        H = H[..., np.newaxis]

    return H


def divergence_kullbackleibler_symmetrised(X, Y=None, cartesian_product=False,
                                           base=2, fill_value=-1,
                                           estimator='ML', Alphabet_X=None,
                                           Alphabet_Y=None, keep_dims=False):
    """
    Returns the symmetrised Kullback-Leibler divergence [Lin91] between arrays
    X and Y, each containing discrete random variable realisations.

    **Mathematical definition**:

    Denoting with :math:`P_X`, :math:`P_Y` respectively probability
    distributions with common domain, associated with discrete random variables
    :math:`X`, :math:`Y`, the symmetrised Kullback-Leibler divergence
    :math:`D_{\\mathrm{SKL}}(P_X \\parallel P_Y)` is defined as:

    .. math::
        D_{\\mathrm{SKL}}(P_X \\parallel P_Y) =
        D_{\\mathrm{KL}}(P_X \\parallel P_Y) +
        D_{\\mathrm{KL}}(P_Y \\parallel P_X)

    where :math:`D_{\\mathrm{KL}}(\\cdot \\parallel \\cdot)` denotes the
    Kullback-Leibler divergence.

    **Estimation**:

    Symmetrised Kullback-Leibler divergence is estimated based on frequency
    tables, using the following functions:

        entropy_cross()

        entropy()

    See below for a list of available estimators. Note that although
    symmetrised Kullback-Leibler divergence is a non-negative quantity,
    depending on the chosen estimator the obtained estimate may be negative.

    **Parameters**:

    X,Y : numpy array (or array-like object such as a list of immutables, as \
    accepted by np.array())
        *cartesian_product==False and Y is not None*: X and Y are arrays
        containing discrete random variable realisations, with
        X.shape[:-1]==Y.shape[:-1]. Successive realisations of a random
        variable are indexed by the last axis in the respective arrays;
        multiple random variables in X and Y may be specified using preceding
        axes of the respective arrays (random variables are paired
        **one-to-one** between X and Y). When X.ndim==Y.ndim==1, returns a
        scalar. When X.ndim>1 and Y.ndim>1, returns an array of estimated
        divergence values with dimensions X.shape[:-1]. Neither X nor Y may
        contain (floating point) NaN values. Missing data may be specified
        using numpy masked arrays, as well as using standard numpy
        array/array-like objects; see below for details.

        *cartesian_product==True and Y is not None*: X and Y are arrays
        containing discrete random variable realisations. Successive
        realisations of a random variable are indexed by the last axis in the
        respective arrays; multiple random variables in X and Y may be
        specified using preceding axes of the respective arrays (random
        variables are paired **many-to-many** between X and Y). When
        X.ndim==Y.ndim==1, returns a scalar. When X.ndim>1 or Y.ndim>1, returns
        an array of estimated divergence values with dimensions
        np.append(X.shape[:-1],Y.shape[:-1]). Neither X nor Y may contain
        (floating point) NaN values. Missing data may be specified using numpy
        masked arrays, as well as using standard numpy array/array-like
        objects; see below for details.

        *Y is None*: Equivalent to divergence_kullbackleibler_symmetrised(X, X,
        ... ). Thus, a shorthand syntax for computing symmetrised
        Kullback-Leibler divergence (in bits) between all pairs of random
        variables in X is divergence_kullbackleibler_symmetrised(X).
    cartesian_product : boolean
        Indicates whether random variables are paired **one-to-one** between X
        and Y (cartesian_product==False, the default value) or **many-to-many**
        between X and Y (cartesian_product==True).
    base : float
        The desired logarithmic base (default 2).
    fill_value : object
        It is possible to specify missing data using numpy masked arrays,
        pandas Series/DataFrames, as well as using standard numpy
        array/array-like objects with assigned placeholder values. When using
        numpy masked arrays, this function invokes np.ma.filled() internally,
        so that missing data are represented with the array's object-internal
        placeholder value fill_value (this function's fill_value parameter is
        ignored in such cases). When using pandas Series/DataFrames, an initial
        conversion to a numpy masked array is performed. When using standard
        numpy array/array-like objects, this function's fill_value parameter is
        used to specify the placeholder value for missing data (defaults to
        -1).

        Data equal to the placeholder value are subsequently ignored.
    estimator : str or float
        The desired estimator (see above for details on estimators). Possible
        values are:

            *'ML' (the default value)* : Maximum likelihood estimator.

            *any floating point value* : Maximum a posteriori esimator using
            Dirichlet prior (equivalent to maximum likelihood with pseudo-count
            for each outcome as specified).

            *PERKS* : Maximum a posteriori esimator using Dirichlet prior
            (equivalent to maximum likelihood with pseudo-count for each
            outcome set to 1/L, where L is the number of possible outcomes.

            *MINIMAX* : Maximum a posteriori esimator using Dirichlet prior
            (equivalent to maximum likelihood with pseudo-count for each
            outcome set to sqrt(N)/L, where N is the total number of
            realisations and where L is the number of possible outcomes.

            *JAMES-STEIN* : James-Stein estimator [HaSt09].

            *GOOD-TURING* : Good-Turing estimator [GaSa95].

    Alphabet_X, Alphabet_Y : numpy array (or array-like object such as a list \
    of immutables, as accepted by np.array())
        Respectively an array specifying the alphabet/alphabets of possible
        outcomes that random variable realisations in array X, Y may assume.
        Defaults to None, in which case the alphabet/alphabets of possible
        outcomes is/are implicitly based the observed outcomes in array X, Y
        respectively, with no additional, unobserved outcomes. In combination
        with any estimator other than maximum likelihood, it may be useful to
        specify alphabets including unobserved outcomes. For such cases,
        successive possible outcomes of a random variable are indexed by the
        last axis in Alphabet_X, Alphabet_Y respectively; multiple alphabets
        may be specified using preceding axes, with the requirement
        X.shape[:-1]==Alphabet_X.shape[:-1] (analogously for Y). Alphabets of
        different sizes may be specified either using numpy masked arrays, or
        by padding with the chosen placeholder fill_value.
    keep_dims : boolean
        When set to True and cartesian_product==False an additional dimension
        of length one is appended to the returned array, facilitating any
        broadcast operations required by the user (defaults to False). Has no
        effect when cartesian_product==True.

    **Implementation notes**:

    Before estimation, outcomes are mapped to the set of non-negative integers
    internally, with the value -1 representing missing data. To avoid this
    internal conversion step, supply integer data and use the default fill
    value -1.

    """
    if Y is None:
        Y = X
        cartesian_product = True
        Alphabet_Y = Alphabet_X

    H1 = divergence_kullbackleibler(X, Y, cartesian_product, base, fill_value,
                                    estimator, Alphabet_X, Alphabet_Y)
    H2 = divergence_kullbackleibler(Y, X, cartesian_product, base, fill_value,
                                    estimator, Alphabet_Y, Alphabet_X)

    if cartesian_product:
        H2 = H2.T

    H = H1 + H2

    if keep_dims and not cartesian_product:
        H = H[..., np.newaxis]

    return H


def entropy_conditional(X, Y=None, cartesian_product=False, base=2,
                        fill_value=-1, estimator='ML', Alphabet_X=None,
                        Alphabet_Y=None, keep_dims=False):
    """
    Returns the conditional entropy (see e.g. [CoTh06]) between arrays X and Y,
    each containing discrete random variable realisations.

    **Mathematical definition**:

    Given discrete random variables :math:`X`, :math:`Y`, the conditional
    entropy :math:`H(X|Y)` is defined as:

    .. math::
        H(X|Y) = H(X,Y) - H(Y)

    where :math:`H(\\cdot,\\cdot)` denotes the joint entropy and where
    :math:`H(\\cdot)` denotes the entropy.

    **Estimation**:

    Conditional entropy is estimated based on frequency tables, using the
    following functions:

        entropy_joint()

        entropy()

    See below for a list of available estimators. Note that although
    conditional entropy is a non-negative quantity, depending on the chosen
    estimator the obtained estimate may be negative.

    **Parameters**:

    X,Y : numpy array (or array-like object such as a list of immutables, as \
    accepted by np.array())
        *cartesian_product==False and Y is not None*: X and Y are arrays
        containing discrete random variable realisations, with
        X.shape==Y.shape. Successive realisations of a random variable are
        indexed by the last axis in the respective arrays; multiple random
        variables in X and Y may be specified using preceding axes of the
        respective arrays (random variables are paired **one-to-one** between X
        and Y). When X.ndim==Y.ndim==1, returns a scalar. When X.ndim>1 and
        Y.ndim>1, returns an array of estimated conditional entropies with
        dimensions X.shape[:-1]. Neither X nor Y may contain (floating point)
        NaN values. Missing data may be specified using numpy masked arrays, as
        well as using standard numpy array/array-like objects; see below
        for details.

        *cartesian_product==True and Y is not None*: X and Y are arrays
        containing discrete random variable realisations, with
        X.shape[-1]==Y.shape[-1]. Successive realisations of a random variable
        are indexed by the last axis in the respective arrays; multiple random
        variables in X and Y may be specified using preceding axes of the
        respective arrays (random variables are paired **many-to-many** between
        X and Y). When X.ndim==Y.ndim==1, returns a scalar. When X.ndim>1 or
        Y.ndim>1, returns an array of estimated conditional entropies with
        dimensions np.append(X.shape[:-1],Y.shape[:-1]). Neither X nor Y may
        contain (floating point) NaN values. Missing data may be specified
        using numpy masked arrays, as well as using standard numpy
        array/array-like objects; see below for details.

        *Y is None*: Equivalent to entropy_conditional(X, X, ... ). Thus, a
        shorthand syntax for computing conditional entropies (in bits) between
        all pairs of random variables in X is entropy_conditional(X).
    cartesian_product : boolean
        Indicates whether random variables are paired **one-to-one** between X
        and Y (cartesian_product==False, the default value) or **many-to-many**
        between X and Y (cartesian_product==True).
    base : float
        The desired logarithmic base (default 2).
    fill_value : object
        It is possible to specify missing data using numpy masked arrays,
        pandas Series/DataFrames, as well as using standard numpy
        array/array-like objects with assigned placeholder values. When using
        numpy masked arrays, this function invokes np.ma.filled() internally,
        so that missing data are represented with the array's object-internal
        placeholder value fill_value (this function's fill_value parameter is
        ignored in such cases). When using pandas Series/DataFrames, an initial
        conversion to a numpy masked array is performed. When using standard
        numpy array/array-like objects, this function's fill_value parameter is
        used to specify the placeholder value for missing data (defaults to
        -1).

        Data equal to the placeholder value are subsequently ignored.
    estimator : str or float
        The desired estimator (see above for details on estimators). Possible
        values are:

            *'ML' (the default value)* : Maximum likelihood estimator.

            *any floating point value* : Maximum a posteriori esimator using
            Dirichlet prior (equivalent to maximum likelihood with pseudo-count
            for each outcome as specified).

            *PERKS* : Maximum a posteriori esimator using Dirichlet prior
            (equivalent to maximum likelihood with pseudo-count for each
            outcome set to 1/L, where L is the number of possible outcomes.

            *MINIMAX* : Maximum a posteriori esimator using Dirichlet prior
            (equivalent to maximum likelihood with pseudo-count for each
            outcome set to sqrt(N)/L, where N is the total number of
            realisations and where L is the number of possible outcomes.

            *JAMES-STEIN* : James-Stein estimator [HaSt09].

            *GOOD-TURING* : Good-Turing estimator [GaSa95].

    Alphabet_X, Alphabet_Y : numpy array (or array-like object such as a list \
    of immutables, as accepted by np.array())
        Respectively an array specifying the alphabet/alphabets of possible
        outcomes that random variable realisations in array X, Y may assume.
        Defaults to None, in which case the alphabet/alphabets of possible
        outcomes is/are implicitly based the observed outcomes in array X, Y
        respectively, with no additional, unobserved outcomes. In combination
        with any estimator other than maximum likelihood, it may be useful to
        specify alphabets including unobserved outcomes. For such cases,
        successive possible outcomes of a random variable are indexed by the
        last axis in Alphabet_X, Alphabet_Y respectively; multiple alphabets
        may be specified using preceding axes, with the requirement
        X.shape[:-1]==Alphabet_X.shape[:-1] (analogously for Y). Alphabets of
        different sizes may be specified either using numpy masked arrays, or
        by padding with the chosen placeholder fill_value.

        NB: When specifying alphabets, an alphabet of possible joint outcomes
        is always implicit from the alphabets of possible (marginal) outcomes
        in Alphabet_X, Alphabet_Y. For example, specifying
        Alphabet_X=Alphabet_Y=np.array(((1,2)) implies an alphabet of possible
        joint outcomes np.array(((1,1,2,2),(1,2,1,2))).
    keep_dims : boolean
        When set to True and cartesian_product==False an additional dimension
        of length one is appended to the returned array, facilitating any
        broadcast operations required by the user (defaults to False). Has no
        effect when cartesian_product==True.

    **Implementation notes**:

    Before estimation, outcomes are mapped to the set of non-negative integers
    internally, with the value -1 representing missing data. To avoid this
    internal conversion step, supply integer data and use the default fill
    value -1.

    """
    # TODO Add note in documentation (for other functions where appropriate)
    # about creating joint observations using appropriate function
    if Y is None:
        Y = X
        cartesian_product = True
        Alphabet_Y = Alphabet_X

    X, fill_value_X = _sanitise_array_input(X, fill_value)
    Y, fill_value_Y = _sanitise_array_input(Y, fill_value)
    if Alphabet_X is not None:
        Alphabet_X, fill_value_Alphabet_X = _sanitise_array_input(Alphabet_X,
                                                                  fill_value)
        Alphabet_X, _ = _autocreate_alphabet(Alphabet_X, fill_value_Alphabet_X)
    else:
        Alphabet_X, fill_value_Alphabet_X = _autocreate_alphabet(X,
                                                                 fill_value_X)
    if Alphabet_Y is not None:
        Alphabet_Y, fill_value_Alphabet_Y = _sanitise_array_input(Alphabet_Y,
                                                                  fill_value)
        Alphabet_Y, _ = _autocreate_alphabet(Alphabet_Y, fill_value_Alphabet_Y)
    else:
        Alphabet_Y, fill_value_Alphabet_Y = _autocreate_alphabet(Y,
                                                                 fill_value_Y)

    if X.size == 0:
        raise ValueError("arg X contains no elements")
    if Y.size == 0:
        raise ValueError("arg Y contains no elements")
    if np.any(_isnan(X)):
        raise ValueError("arg X contains NaN values")
    if np.any(_isnan(Y)):
        raise ValueError("arg Y contains NaN values")
    if Alphabet_X.size == 0:
        raise ValueError("arg Alphabet_X contains no elements")
    if np.any(_isnan(Alphabet_X)):
        raise ValueError("arg Alphabet_X contains NaN values")
    if Alphabet_Y.size == 0:
        raise ValueError("arg Alphabet_Y contains no elements")
    if np.any(_isnan(Alphabet_Y)):
        raise ValueError("arg Alphabet_Y contains NaN values")
    if _isnan(fill_value_X):
        raise ValueError("fill value for arg X is NaN")
    if _isnan(fill_value_Y):
        raise ValueError("fill value for arg Y is NaN")
    if X.shape[:-1] != Alphabet_X.shape[:-1]:
        raise ValueError("leading dimensions of args X and Alphabet_X do not "
                         "match")
    if Y.shape[:-1] != Alphabet_Y.shape[:-1]:
        raise ValueError("leading dimensions of args Y and Alphabet_Y do not "
                         "match")
    if not cartesian_product and X.shape != Y.shape:
        raise ValueError("dimensions of args X and Y do not match")
    if cartesian_product and X.shape[-1] != Y.shape[-1]:
        raise ValueError("trailing dimensions of args X and Y do not match")
    if not (np.isscalar(base) and np.isreal(base) and base > 0):
        raise ValueError("arg base not a positive real-valued scalar")

    S, fill_value = _map_observations_to_integers((X, Alphabet_X,
                                                   Y, Alphabet_Y),
                                                  (fill_value_X,
                                                   fill_value_Alphabet_X,
                                                   fill_value_Y,
                                                   fill_value_Alphabet_Y))
    X, Alphabet_X, Y, Alphabet_Y = S

    if not cartesian_product:
        H = np.empty(X.shape[:-1])
        if H.ndim > 0:
            H[:] = np.NaN
        else:
            H = np.float64(np.NaN)
    else:
        def f(X, Y, Alphabet_X, Alphabet_Y):
            return entropy_conditional(X, Y, False, base, fill_value,
                                       estimator, Alphabet_X, Alphabet_Y)
        return _cartesian_product_apply(X, Y, f, Alphabet_X, Alphabet_Y)

    # Re-shape H, X and Y, so that we may handle multi-dimensional arrays
    # equivalently and iterate across 0th axis
    X = np.reshape(X, (-1, X.shape[-1]))
    Y = np.reshape(Y, (-1, Y.shape[-1]))
    Alphabet_X = np.reshape(Alphabet_X, (-1, Alphabet_X.shape[-1]))
    Alphabet_Y = np.reshape(Alphabet_Y, (-1, Alphabet_Y.shape[-1]))
    orig_shape_H = H.shape
    H = np.reshape(H, (-1, 1))

    for i in range(X.shape[0]):
        H[i] = entropy_joint(np.vstack((X[i], Y[i])), base, fill_value,
                             estimator, _vstack_pad((Alphabet_X[i],
                                                     Alphabet_Y[i]),
                                                    fill_value)) - \
            entropy(Y[i], base, fill_value, estimator, Alphabet_Y[i])

    # Reverse re-shaping
    H = np.reshape(H, orig_shape_H)

    if keep_dims and not cartesian_product:
        H = H[..., np.newaxis]

    return H


def entropy_joint(X, base=2, fill_value=-1, estimator='ML', Alphabet_X=None,
                  keep_dims=False):
    """
    Returns the estimated joint entropy (see e.g. [CoTh06]) for an array X
    containing realisations of discrete random variables.

    **Mathematical definition**:

    Denoting with :math:`P(x_1, \\ldots, x_n)` the probability of jointly
    observing outcomes :math:`(x_1, \\ldots, x_n)` of :math:`n` discrete random
    variables :math:`(X_1, \ldots, X_n)`, the joint entropy
    :math:`H(X_1, \\ldots, X_n)` is defined as:

    .. math::
        H(X_1, \\ldots, X_n) = -\\sum_{x_1} \\ldots \\sum_{x_n}
        {P(x_1, \\ldots, x_n ) \\log {P(x_1, \\ldots, x_n)}}.

    **Estimation**:

    Joint entropy is estimated based on frequency tables. See below for a list
    of available estimators.

    *Parameters**:

    X : numpy array (or array-like object such as a list of immutables, as \
    accepted by np.array())
        An array containing discrete random variable realisations. Successive
        realisations of a random variable are indexed by the last axis in the
        array; multiple random variables may be specified using preceding axes.
        When X.ndim==1, returns a scalar and is equivalent to entropy(). When
        X.ndim>1, returns a scalar based on jointly considering all random
        variables indexed in the array. X may not contain (floating point) NaN
        values. Missing data may be specified using numpy masked arrays, as
        well as using standard numpy array/array-like objects; see below
        for details.
    base : float
        The desired logarithmic base (default 2).
    fill_value : object
        It is possible to specify missing data using numpy masked arrays,
        pandas Series/DataFrames, as well as using standard numpy
        array/array-like objects with assigned placeholder values. When using
        numpy masked arrays, this function invokes np.ma.filled() internally,
        so that missing data are represented with the array's object-internal
        placeholder value fill_value (this function's fill_value parameter is
        ignored in such cases). When using pandas Series/DataFrames, an initial
        conversion to a numpy masked array is performed. When using standard
        numpy array/array-like objects, this function's fill_value parameter is
        used to specify the placeholder value for missing data (defaults to
        -1).

        Data equal to the placeholder value are subsequently ignored.
    estimator : str or float
        The desired estimator (see above for details on estimators). Possible
        values are:

            *'ML' (the default value)* : Maximum likelihood estimator.

            *any floating point value* : Maximum a posteriori esimator using
            Dirichlet prior (equivalent to maximum likelihood with pseudo-count
            for each outcome as specified).

            *PERKS* : Maximum a posteriori esimator using Dirichlet prior
            (equivalent to maximum likelihood with pseudo-count for each
            outcome set to 1/L, where L is the number of possible outcomes.

            *MINIMAX* : Maximum a posteriori esimator using Dirichlet prior
            (equivalent to maximum likelihood with pseudo-count for each
            outcome set to sqrt(N)/L, where N is the total number of
            realisations and where L is the number of possible outcomes.

            *JAMES-STEIN* : James-Stein estimator [HaSt09].

            *GOOD-TURING* : Good-Turing estimator [GaSa95].

    Alphabet_X : numpy array (or array-like object such as a list of \
    immutables, as accepted by np.array())
        An array specifying the alphabet/alphabets of possible outcomes that
        random variable realisations in array X may assume. Defaults to None,
        in which case the alphabet/alphabets of possible outcomes is/are
        implicitly based the observed outcomes in array X, with no additional,
        unobserved outcomes. In combination with any estimator other than
        maximum likelihood, it may be useful to specify alphabets including
        unobserved outcomes. For such cases, successive possible outcomes of a
        random variable are indexed by the last axis in Alphabet_X; multiple
        alphabets may be specified using preceding axes, with the requirement
        X.shape[:-1]==Alphabet_X.shape[:-1]. Alphabets of different sizes may
        be specified either using numpy masked arrays, or by padding with the
        chosen placeholder fill_value.

        NB: When specifying multiple alphabets, an alphabet of possible joint
        outcomes is always implicit from the alphabets of possible (marginal)
        outcomes in Alphabet_X. For example, specifying
        Alphabet_X=np.array(((1,2),(1,2))) implies an alphabet of possible
        joint outcomes np.array(((1,1,2,2),(1,2,1,2))).
    keep_dims : boolean
        When set to True, an additional dimension of length one is appended to
        the returned array, facilitating any broadcast operations required by
        the user (defaults to False).

    **Implementation notes**:

    Before estimation, outcomes are mapped to the set of non-negative integers
    internally, with the value -1 representing missing data. To avoid this
    internal conversion step, supply integer data and use the default fill
    value -1.

    """
    # TODO If we add joint observation function, we can reduce code duplication
    # in this function.
    # TODO NB: The joint observation function must honour missing data fill
    # values.
    X, fill_value_X = _sanitise_array_input(X, fill_value)
    if Alphabet_X is not None:
        Alphabet_X, fill_value_Alphabet_X = _sanitise_array_input(Alphabet_X,
                                                                  fill_value)
        Alphabet_X, _ = _autocreate_alphabet(Alphabet_X, fill_value_Alphabet_X)
    else:
        Alphabet_X, fill_value_Alphabet_X = _autocreate_alphabet(X,
                                                                 fill_value_X)

    if X.size == 0:
        raise ValueError("arg X contains no elements")
    if np.any(_isnan(X)):
        raise ValueError("arg X contains NaN values")
    if Alphabet_X.size == 0:
        raise ValueError("arg Alphabet_X contains no elements")
    if np.any(_isnan(Alphabet_X)):
        raise ValueError("arg Alphabet_X contains NaN values")
    if _isnan(fill_value_X):
        raise ValueError("fill value for arg X is NaN")
    if X.shape[:-1] != Alphabet_X.shape[:-1]:
        raise ValueError("leading dimensions of args X and Alphabet_X do not "
                         "match")
    if not (np.isscalar(base) and np.isreal(base) and base > 0):
        raise ValueError("arg base not a positive real-valued scalar")

    S, fill_value = _map_observations_to_integers((X, Alphabet_X),
                                                  (fill_value_X,
                                                   fill_value_Alphabet_X))
    X, Alphabet_X = S

    # Re-shape X, so that we may handle multi-dimensional arrays equivalently
    # and iterate across 0th axis
    X = np.reshape(X, (-1, X.shape[-1]))
    Alphabet_X = np.reshape(Alphabet_X, (-1, Alphabet_X.shape[-1]))

    _verify_alphabet_sufficiently_large(X, Alphabet_X, fill_value)

    # Sort columns
    for i in range(X.shape[0]):
        X = X[:, X[i].argsort(kind='mergesort')]

    # Compute symbol run-lengths
    # Compute symbol change indicators
    B = np.any(X[:, 1:] != X[:, :-1], axis=0)
    # Obtain symbol change positions
    I = np.append(np.where(B), X.shape[1]-1)
    # Compute run lengths
    L = np.diff(np.append(-1, I))

    alphabet_X = X[:, I]
    if estimator != 'ML':
        n_additional_empty_bins = \
            _determine_number_additional_empty_bins(L, alphabet_X, Alphabet_X,
                                                    fill_value)
    else:
        n_additional_empty_bins = 0
    L, _ = _remove_counts_at_fill_value(L, alphabet_X, fill_value)
    if not np.any(L):
        return np.float64(np.NaN)

    # P_0 is the probability mass assigned to each additional empty bin
    P, P_0 = _estimate_probabilities(L, estimator, n_additional_empty_bins)
    H_0 = n_additional_empty_bins * P_0 * \
        -np.log2(P_0 + np.spacing(0)) / np.log2(base)
    H = entropy_pmf(P, base, require_valid_pmf=False) + H_0

    if keep_dims:
        H = H[..., np.newaxis]

    return H


def entropy(X, base=2, fill_value=-1, estimator='ML', Alphabet_X=None,
            keep_dims=False):
    """
    Returns the estimated entropy (see e.g. [CoTh06]) for an array X containing
    realisations of a discrete random variable.

    **Mathematical definition**:

    Denoting with :math:`P(x)` the probability of observing outcome :math:`x`
    of a discrete random variable :math:`X`, the entropy :math:`H(X)` is
    defined as:

    .. math::
        H(X) = -\\sum_x {P(x) \\log {P(x)}}.

    **Estimation**:

    Entropy is estimated based on frequency tables. See below for a list of
    available estimators.

    **Parameters**:

    X : numpy array (or array-like object such as a list of immutables, as \
    accepted by np.array())
        An array containing discrete random variable realisations. Successive
        realisations of a random variable are indexed by the last axis in the
        array; multiple random variables may be specified using preceding axes.
        When X.ndim==1, returns a scalar. When X.ndim>1, returns an array of
        estimated entropies with dimensions X.shape[:-1].  X may not contain
        (floating point) NaN values. Missing data may be specified using numpy
        masked arrays, as well as using standard numpy array/array-like
        objects; see below for details.
    base : float
        The desired logarithmic base (default 2).
    fill_value : object
        It is possible to specify missing data using numpy masked arrays,
        pandas Series/DataFrames, as well as using standard numpy
        array/array-like objects with assigned placeholder values. When using
        numpy masked arrays, this function invokes np.ma.filled() internally,
        so that missing data are represented with the array's object-internal
        placeholder value fill_value (this function's fill_value parameter is
        ignored in such cases). When using pandas Series/DataFrames, an initial
        conversion to a numpy masked array is performed. When using standard
        numpy array/array-like objects, this function's fill_value parameter is
        used to specify the placeholder value for missing data (defaults to
        -1).

        Data equal to the placeholder value are subsequently ignored.
    estimator : str or float
        The desired estimator (see above for details on estimators). Possible
        values are:

            *'ML' (the default value)* : Maximum likelihood estimator.

            *any floating point value* : Maximum a posteriori esimator using
            Dirichlet prior (equivalent to maximum likelihood with pseudo-count
            for each outcome as specified).

            *PERKS* : Maximum a posteriori esimator using Dirichlet prior
            (equivalent to maximum likelihood with pseudo-count for each
            outcome set to 1/L, where L is the number of possible outcomes.

            *MINIMAX* : Maximum a posteriori esimator using Dirichlet prior
            (equivalent to maximum likelihood with pseudo-count for each
            outcome set to sqrt(N)/L, where N is the total number of
            realisations and where L is the number of possible outcomes.

            *JAMES-STEIN* : James-Stein estimator [HaSt09].

            *GOOD-TURING* : Good-Turing estimator [GaSa95].

    Alphabet_X : numpy array (or array-like object such as a list of \
    immutables, as accepted by np.array())
        An array specifying the alphabet/alphabets of possible outcomes that
        random variable realisations in array X may assume. Defaults to None,
        in which case the alphabet/alphabets of possible outcomes is/are
        implicitly based the observed outcomes in array X, with no additional,
        unobserved outcomes. In combination with any estimator other than
        maximum likelihood, it may be useful to specify alphabets including
        unobserved outcomes. For such cases, successive possible outcomes of a
        random variable are indexed by the last axis in Alphabet_X; multiple
        alphabets may be specified using preceding axes, with the requirement
        X.shape[:-1]==Alphabet_X.shape[:-1]. Alphabets of different sizes may
        be specified either using numpy masked arrays, or by padding with the
        chosen placeholder fill_value.
    keep_dims : boolean
        When set to True, an additional dimension of length one is appended to
        the returned array, facilitating any broadcast operations required by
        the user (defaults to False).

    **Implementation notes**:

    Before estimation, outcomes are mapped to the set of non-negative integers
    internally, with the value -1 representing missing data. To avoid this
    internal conversion step, supply integer data and use the default fill
    value -1.

    """
    # NB: We would be able to reduce code duplication by invoking
    # entropy_cross(X,X). However, performance would likely be lower!
    X, fill_value_X = _sanitise_array_input(X, fill_value)
    if Alphabet_X is not None:
        Alphabet_X, fill_value_Alphabet_X = _sanitise_array_input(Alphabet_X,
                                                                  fill_value)
        Alphabet_X, _ = _autocreate_alphabet(Alphabet_X, fill_value_Alphabet_X)
    else:
        Alphabet_X, fill_value_Alphabet_X = _autocreate_alphabet(X,
                                                                 fill_value_X)

    if X.size == 0:
        raise ValueError("arg X contains no elements")
    if np.any(_isnan(X)):
        raise ValueError("arg X contains NaN values")
    if Alphabet_X.size == 0:
        raise ValueError("arg Alphabet_X contains no elements")
    if np.any(_isnan(Alphabet_X)):
        raise ValueError("arg Alphabet_X contains NaN values")
    if _isnan(fill_value_X):
        raise ValueError("fill value for arg X is NaN")
    if X.shape[:-1] != Alphabet_X.shape[:-1]:
        raise ValueError("leading dimensions of args X and Alphabet_X do not "
                         "match")
    if not (np.isscalar(base) and np.isreal(base) and base > 0):
        raise ValueError("arg base not a positive real-valued scalar")

    S, fill_value = _map_observations_to_integers((X, Alphabet_X),
                                                  (fill_value_X,
                                                   fill_value_Alphabet_X))
    X, Alphabet_X = S

    H = np.empty(X.shape[:-1])
    if H.ndim > 0:
        H[:] = np.NaN
    else:
        H = np.float64(np.NaN)

    # Re-shape H and X, so that we may handle multi-dimensional arrays
    # equivalently and iterate across 0th axis
    X = np.reshape(X, (-1, X.shape[-1]))
    Alphabet_X = np.reshape(Alphabet_X, (-1, Alphabet_X.shape[-1]))
    orig_shape_H = H.shape
    H = np.reshape(H, (-1, 1))

    _verify_alphabet_sufficiently_large(X, Alphabet_X, fill_value)

    # NB: This is not joint entropy. Elements in each row are sorted
    # independently
    X = np.sort(X, axis=1)

    # Compute symbol run-lengths
    # Compute symbol change indicators
    B = X[:, 1:] != X[:, :-1]
    for i in range(X.shape[0]):
        # Obtain symbol change positions
        I = np.append(np.where(B[i]), X.shape[1]-1)
        # Compute run lengths
        L = np.diff(np.append(-1, I))

        alphabet_X = X[i, I]
        if estimator != 'ML':
            n_additional_empty_bins = \
                _determine_number_additional_empty_bins(L, alphabet_X,
                                                        Alphabet_X[i],
                                                        fill_value)
        else:
            n_additional_empty_bins = 0
        L, _ = _remove_counts_at_fill_value(L, alphabet_X, fill_value)
        if not np.any(L):
            continue

        # P_0 is the probability mass assigned to each additional empty bin
        P, P_0 = _estimate_probabilities(L, estimator, n_additional_empty_bins)
        H_0 = n_additional_empty_bins * P_0 * \
            -np.log2(P_0 + np.spacing(0)) / np.log2(base)
        H[i] = entropy_pmf(P, base, require_valid_pmf=False) + H_0

    # Reverse re-shaping
    H = np.reshape(H, orig_shape_H)

    if keep_dims:
        H = H[..., np.newaxis]

    return H


def entropy_pmf(P, base=2, require_valid_pmf=True, keep_dims=False):
    """
    Returns the entropy (see e.g. [CoTh06]) of an array P representing a
    discrete probability distribution.

    **Mathematical definition**:

    Denoting with :math:`P(x)` the probability mass associated with observing
    an outcome :math:`x` under distribution :math:`P`, the entropy :math:`H(P)`
    is defined as:

    .. math::
        H(P) = -\\sum_x {P(x) \\log {P(x)}}.

    **Parameters**:

    P : numpy array (or array-like object such as a list of immutables, as \
    accepted by np.array())
        An array containing probability mass assignments. Probabilities in a
        distribution are indexed by the last axis in the array; multiple
        probability distributions may be specified using preceding axes. When
        P.ndim==1, returns a scalar. When P.ndim>1, returns an array of
        entropies with dimensions P.shape[:-1]. P may not contain (floating
        point) NaN values.
    base : float
        The desired logarithmic base (default 2).
    require_valid_pmf : boolean
        When set to True (the default value), verifies that probability mass
        assignments in each distribution sum to 1. When set to False, no such
        test is performed, thus allowing incomplete probability distributions
        to be processed.
    keep_dims : boolean
        When set to True, an additional dimension of length one is appended to
        the returned array, facilitating any broadcast operations required by
        the user (defaults to False).
    """
    P, _ = _sanitise_array_input(P)

    if P.size == 0:
        raise ValueError("arg P contains no elements")
    if np.any(_isnan(P)):
        raise ValueError("arg P contains NaN values")
    if np.any(np.logical_or(P < 0, P > 1)):
        raise ValueError("arg P contains values outside unit interval")
    if require_valid_pmf and not np.allclose(np.sum(P, axis=-1), 1):
        raise ValueError("arg P does not sum to unity across last axis")
    if not (np.isscalar(base) and np.isreal(base) and base > 0):
        raise ValueError("arg base not a positive real-valued scalar")

    H = -np.sum(P * np.log2(P + np.spacing(0)), axis=-1)
    H = H / np.log2(base)

    if keep_dims:
        H = H[..., np.newaxis]

    return H


def entropy_cross_pmf(P, Q=None, cartesian_product=False, base=2,
                      require_valid_pmf=True, keep_dims=False):
    """
    Returns the cross entropy (see e.g. [Murp12]) between arrays P and Q, each
    representing a discrete probability distribution.

    **Mathematical definition**:

    Denoting with :math:`P(x)`, :math:`Q(x)` respectively the probability mass
    associated with observing an outcome :math:`x` under distributions
    :math:`P`, :math:`Q`, the cross entropy :math:`H^\\times(P,Q)` is defined
    as:

    .. math::
        H^\\times(P,Q) = -\\sum_x {P(x) \\log {Q(x)}}.

    **Parameters**:

    P, Q : numpy array (or array-like object such as a list of immutables, as \
    accepted by np.array())
        *cartesian_product==False and Q is not None*: P and Q are arrays
        containing probability mass assignments, with P.shape==Q.shape.
        Probabilities in a distribution are indexed by the last axis in the
        respective arrays; multiple probability distributions in P and Q may be
        specified using preceding axes of the respective arrays (distributions
        are paired **one-to-one** between P and Q). When P.ndim==Q.ndim==1,
        returns a scalar. When P.ndim>1 and Q.ndim>1, returns an array of cross
        entropies with dimensions P.shape[:-1]. Neither P nor Q may contain
        (floating point) NaN values.

        *cartesian_product==True and Q is not None*: P and Q are arrays
        containing probability mass assignments, with P.shape[-1]==Q.shape[-1].
        Probabilities in a distribution are indexed by the last axis in the
        respective arrays; multiple probability distributions in P and Q may be
        specified using preceding axes of the respective arrays (distributions
        are paired **many-to-many** between P and Q). When P.ndim==Q.ndim==1,
        returns a scalar. When P.ndim>1 and Q.ndim>1, returns an array of cross
        entropies with dimensions np.append(P.shape[:-1],Q.shape[:-1]). Neither
        P nor Q may contain (floating point) NaN values.

        *Q is None*: Equivalent to entropy_cross_pmf(P, P, ... ). Thus, a
        shorthand syntax for computing cross entropies (in bits) between all
        pairs of probability distributions in P is entropy_cross_pmf(P).
    cartesian_product : boolean
        Indicates whether probability distributions are paired **one-to-one**
        between P and Q (cartesian_product==False, the default value) or
        **many-to-many** between P and Q (cartesian_product==True).
    base : float
        The desired logarithmic base (default 2).
    require_valid_pmf : boolean
        When set to True (the default value), verifies that probability mass
        assignments in each distribution sum to 1. When set to False, no such
        test is performed, thus allowing incomplete probability distributions
        to be processed.
    keep_dims : boolean
        When set to True and cartesian_product==False an additional dimension
        of length one is appended to the returned array, facilitating any
        broadcast operations required by the user (defaults to False). Has no
        effect when cartesian_product==True.
    """
    if Q is None:
        Q = P
        cartesian_product = True

    P, _ = _sanitise_array_input(P)
    Q, _ = _sanitise_array_input(Q)

    if P.size == 0:
        raise ValueError("arg P contains no elements")
    if Q.size == 0:
        raise ValueError("arg Q contains no elements")
    if np.any(_isnan(P)):
        raise ValueError("arg P contains NaN values")
    if np.any(_isnan(Q)):
        raise ValueError("arg Q contains NaN values")
    if not cartesian_product and P.shape != Q.shape:
        raise ValueError("dimensions of args P and Q do not match")
    if cartesian_product and P.shape[-1] != Q.shape[-1]:
        raise ValueError("trailing dimensions of args P and Q do not match")
    if np.any(np.logical_or(P < 0, P > 1)):
        raise ValueError("arg P contains values outside unit interval")
    if np.any(np.logical_or(Q < 0, Q > 1)):
        raise ValueError("arg Q contains values outside unit interval")
    if require_valid_pmf and not np.allclose(np.sum(P, axis=-1), 1):
        raise ValueError("arg P does not sum to unity across last axis")
    if require_valid_pmf and not np.allclose(np.sum(Q, axis=-1), 1):
        raise ValueError("arg Q does not sum to unity across last axis")
    if not (np.isscalar(base) and np.isreal(base) and base > 0):
        raise ValueError("arg base not a positive real-valued scalar")

    if cartesian_product:
        def f(P, Q):
            return entropy_cross_pmf(P, Q, False, base, require_valid_pmf)
        return _cartesian_product_apply(P, Q, f)

    with np.errstate(invalid='ignore', divide='ignore'):
        H = P * np.log2(Q)
    H[P == 0] = 0
    H = -np.sum(H, axis=-1)
    H = H / np.log2(base)

    if keep_dims and not cartesian_product:
        H = H[..., np.newaxis]

    return H


def divergence_kullbackleibler_pmf(P, Q=None, cartesian_product=False, base=2,
                                   require_valid_pmf=True, keep_dims=False):
    """
    Returns the Kullback-Leibler divergence (see e.g. [CoTh06]) between arrays
    P and Q, each representing a discrete probability distribution.

    **Mathematical definition**:

    Denoting with :math:`P(x)`, :math:`Q(x)` respectively the probability mass
    associated with observing an outcome :math:`x` under distributions
    :math:`P`, :math:`Q`, the Kullback-Leibler divergence
    :math:`D_{\\mathrm{KL}}(P \\parallel Q)` is defined as:

    .. math::
        D_{\\mathrm{KL}}(P \\parallel Q) =
        -\\sum_x {P(x) \\log {\\frac{Q(x)}{P(x)}}}.

    **Parameters**:

    P, Q : numpy array (or array-like object such as a list of immutables, as \
    accepted by np.array())
        *cartesian_product==False and Q is not None*: P and Q are arrays
        containing probability mass assignments, with P.shape==Q.shape.
        Probabilities in a distribution are indexed by the last axis in the
        respective arrays; multiple probability distributions in P and Q may be
        specified using preceding axes of the respective arrays (distributions
        are paired **one-to-one** between P and Q). When P.ndim==Q.ndim==1,
        returns a scalar. When P.ndim>1 and Q.ndim>1, returns an array of
        divergence values with dimensions P.shape[:-1]. Neither P nor Q may
        contain (floating point) NaN values.

        *cartesian_product==True and Q is not None*: P and Q are arrays
        containing probability mass assignments, with P.shape[-1]==Q.shape[-1].
        Probabilities in a distribution are indexed by the last axis in the
        respective arrays; multiple probability distributions in P and Q may be
        specified using preceding axes of the respective arrays (distributions
        are paired **many-to-many** between P and Q). When P.ndim==Q.ndim==1,
        returns a scalar. When P.ndim>1 and Q.ndim>1, returns an array of
        divergence values with dimensions np.append(P.shape[:-1],Q.shape[:-1]).
        Neither P nor Q may contain (floating point) NaN values.

        *Q is None*: Equivalent to divergence_kullbackleibler_pmf(P, P, ... ).
        Thus, a shorthand syntax for computing Kullback-Leibler divergence (in
        bits) between all pairs of probability distributions in P is
        divergence_kullbackleibler_pmf(P).
    cartesian_product : boolean
        Indicates whether probability distributions are paired **one-to-one**
        between P and Q (cartesian_product==False, the default value) or
        **many-to-many** between P and Q (cartesian_product==True).
    base : float
        The desired logarithmic base (default 2).
    require_valid_pmf : boolean
        When set to True (the default value), verifies that probability mass
        assignments in each distribution sum to 1. When set to False, no such
        test is performed, thus allowing incomplete probability distributions
        to be processed.
    keep_dims : boolean
        When set to True and cartesian_product==False an additional dimension
        of length one is appended to the returned array, facilitating any
        broadcast operations required by the user (defaults to False). Has no
        effect when cartesian_product==True.
    """
    H_cross = entropy_cross_pmf(P, Q, cartesian_product, base,
                                require_valid_pmf)
    H = entropy_pmf(P, base, require_valid_pmf)

    H = np.reshape(H, np.append(H.shape,
                                np.ones(H_cross.ndim-H.ndim)).astype('int'))

    H = H_cross - H

    if keep_dims and not cartesian_product:
        H = H[..., np.newaxis]

    return H


def divergence_jensenshannon_pmf(P, Q=None, cartesian_product=False, base=2,
                                 require_valid_pmf=True, keep_dims=False):
    """
    Returns the Jensen-Shannon divergence [Lin91] between arrays P and Q, each
    representing a discrete probability distribution.

    **Mathematical definition**:

    Denoting with :math:`P`, :math:`Q` probability distributions with common
    domain, the Jensen-Shannon divergence
    :math:`D_{\\mathrm{JS}}(P \\parallel Q)` is defined as:

    .. math::
        D_{\\mathrm{JS}}(P \\parallel Q) =
        \\frac{1}{2} D_{\\mathrm{KL}}(P \\parallel M) +
        \\frac{1}{2} D_{\\mathrm{KL}}(Q \\parallel M)

    where :math:`M = \\frac{1}{2}(P + Q)` and where
    :math:`D_{\\mathrm{KL}}(\\cdot \\parallel \\cdot)` denotes the
    Kullback-Leibler divergence.

    **Parameters**:

    P, Q : numpy array (or array-like object such as a list of immutables, as \
    accepted by np.array())
        *cartesian_product==False and Q is not None*: P and Q are arrays
        containing probability mass assignments, with P.shape==Q.shape.
        Probabilities in a distribution are indexed by the last axis in the
        respective arrays; multiple probability distributions in P and Q may be
        specified using preceding axes of the respective arrays (distributions
        are paired **one-to-one** between P and Q). When P.ndim==Q.ndim==1,
        returns a scalar. When P.ndim>1 and Q.ndim>1, returns an array of
        divergence values with dimensions P.shape[:-1]. Neither P nor Q may
        contain (floating point) NaN values.

        *cartesian_product==True and Q is not None*: P and Q are arrays
        containing probability mass assignments, with P.shape[-1]==Q.shape[-1].
        Probabilities in a distribution are indexed by the last axis in the
        respective arrays; multiple probability distributions in P and Q may be
        specified using preceding axes of the respective arrays (distributions
        are paired **many-to-many** between P and Q). When P.ndim==Q.ndim==1,
        returns a scalar. When P.ndim>1 and Q.ndim>1, returns an array of
        divergence values with dimensions np.append(P.shape[:-1],Q.shape[:-1]).
        Neither P nor Q may contain (floating point) NaN values.

        *Q is None*: Equivalent to divergence_jensenshannon_pmf(P, P, ... ).
        Thus, a shorthand syntax for computing Jensen-Shannon divergence (in
        bits) between all pairs of probability distributions in P is
        divergence_jensenshannon_pmf(P).
    cartesian_product : boolean
        Indicates whether probability distributions are paired **one-to-one**
        between P and Q (cartesian_product==False, the default value) or
        **many-to-many** between P and Q (cartesian_product==True).
    base : float
        The desired logarithmic base (default 2).
    require_valid_pmf : boolean
        When set to True (the default value), verifies that probability mass
        assignments in each distribution sum to 1. When set to False, no such
        test is performed, thus allowing incomplete probability distributions
        to be processed.
    keep_dims : boolean
        When set to True and cartesian_product==False an additional dimension
        of length one is appended to the returned array, facilitating any
        broadcast operations required by the user (defaults to False). Has no
        effect when cartesian_product==True.
    """
    if Q is None:
        Q = P
        cartesian_product = True

    P, _ = _sanitise_array_input(P)
    Q, _ = _sanitise_array_input(Q)

    if P.size == 0:
        raise ValueError("arg P contains no elements")
    if Q.size == 0:
        raise ValueError("arg Q contains no elements")
    if np.any(_isnan(P)):
        raise ValueError("arg P contains NaN values")
    if np.any(_isnan(Q)):
        raise ValueError("arg Q contains NaN values")
    if not cartesian_product and P.shape != Q.shape:
        raise ValueError("dimensions of args P and Q do not match")
    if cartesian_product and P.shape[-1] != Q.shape[-1]:
        raise ValueError("trailing dimensions of args P and Q do not match")
    if np.any(np.logical_or(P < 0, P > 1)):
        raise ValueError("arg P contains values outside unit interval")
    if np.any(np.logical_or(Q < 0, Q > 1)):
        raise ValueError("arg Q contains values outside unit interval")
    if require_valid_pmf and not np.allclose(np.sum(P, axis=-1), 1):
        raise ValueError("arg P does not sum to unity across last axis")
    if require_valid_pmf and not np.allclose(np.sum(Q, axis=-1), 1):
        raise ValueError("arg Q does not sum to unity across last axis")
    if not (np.isscalar(base) and np.isreal(base) and base > 0):
        raise ValueError("arg base not a positive real-valued scalar")

    if cartesian_product:
        def f(P, Q):
            return divergence_jensenshannon_pmf(P, Q, False, base,
                                                require_valid_pmf)
        return _cartesian_product_apply(P, Q, f)

    H1 = entropy_pmf(0.5*(P + Q), base, require_valid_pmf)
    H = H1 - 0.5*entropy_pmf(P, base, require_valid_pmf) - \
        0.5*entropy_pmf(Q, base, require_valid_pmf)

    if keep_dims and not cartesian_product:
        H = H[..., np.newaxis]

    return H


def divergence_kullbackleibler_symmetrised_pmf(P, Q=None,
                                               cartesian_product=False, base=2,
                                               require_valid_pmf=True,
                                               keep_dims=False):
    """
    Returns the symmetrised Kullback-Leibler divergence [Lin91] between arrays
    P and Q, each representing a discrete probability distribution.

    **Mathematical definition**:

    Denoting with :math:`P`, :math:`Q` probability distributions with common
    domain, the symmetrised Kullback-Leibler divergence
    :math:`D_{\\mathrm{SKL}}(P \\parallel Q)` is defined as:

    .. math::
        D_{\\mathrm{SKL}}(P \\parallel Q) =
        D_{\\mathrm{KL}}(P \\parallel Q) +
        D_{\\mathrm{KL}}(Q \\parallel P)

    where :math:`D_{\\mathrm{KL}}(\\cdot \\parallel \\cdot)` denotes the
    Kullback-Leibler divergence.

    **Parameters**:

    P, Q : numpy array (or array-like object such as a list of immutables, as \
    accepted by np.array())
        *cartesian_product==False and Q is not None*: P and Q are arrays
        containing probability mass assignments, with P.shape==Q.shape.
        Probabilities in a distribution are indexed by the last axis in the
        respective arrays; multiple probability distributions in P and Q may be
        specified using preceding axes of the respective arrays (distributions
        are paired **one-to-one** between P and Q). When P.ndim==Q.ndim==1,
        returns a scalar. When P.ndim>1 and Q.ndim>1, returns an array of
        divergence values with dimensions P.shape[:-1]. Neither P nor Q may
        contain (floating point) NaN values.

        *cartesian_product==True and Q is not None*: P and Q are arrays
        containing probability mass assignments, with P.shape[-1]==Q.shape[-1].
        Probabilities in a distribution are indexed by the last axis in the
        respective arrays; multiple probability distributions in P and Q may be
        specified using preceding axes of the respective arrays (distributions
        are paired **many-to-many** between P and Q). When P.ndim==Q.ndim==1,
        returns a scalar. When P.ndim>1 and Q.ndim>1, returns an array of
        divergence values with dimensions np.append(P.shape[:-1],Q.shape[:-1]).
        Neither P nor Q may contain (floating point) NaN values.

        *Q is None*: Equivalent to
        divergence_kullbackleibler_symmetrised_pmf(P, P, ... ). Thus, a
        shorthand syntax for computing symmetrised Kullback-Leibler divergence
        (in bits) between all pairs of probability distributions in P is
        divergence_kullbackleibler_symmetrised_pmf(P).
    cartesian_product : boolean
        Indicates whether probability distributions are paired **one-to-one**
        between P and Q (cartesian_product==False, the default value) or
        **many-to-many** between P and Q (cartesian_product==True).
    base : float
        The desired logarithmic base (default 2).
    require_valid_pmf : boolean
        When set to True (the default value), verifies that probability mass
        assignments in each distribution sum to 1. When set to False, no such
        test is performed, thus allowing incomplete probability distributions
        to be processed.
    keep_dims : boolean
        When set to True and cartesian_product==False an additional dimension
        of length one is appended to the returned array, facilitating any
        broadcast operations required by the user (defaults to False). Has no
        effect when cartesian_product==True.
    """
    if Q is None:
        Q = P
        cartesian_product = True

    H1 = divergence_kullbackleibler_pmf(P, Q, cartesian_product, base,
                                        require_valid_pmf)
    H2 = divergence_kullbackleibler_pmf(Q, P, cartesian_product, base,
                                        require_valid_pmf)

    if cartesian_product:
        H2 = H2.T

    H = H1 + H2

    if keep_dims and not cartesian_product:
        H = H[..., np.newaxis]

    return H


def _append_empty_bins_using_alphabet(Counts, Alphabet, Full_Alphabet,
                                      fill_value):
    if Alphabet.ndim == 1:
        assert(Full_Alphabet.ndim == 1)
        A = np.setdiff1d(Full_Alphabet[Full_Alphabet != fill_value], Alphabet,
                         assume_unique=True)
        Alphabet = np.append(Alphabet, A)
        Counts = np.append(Counts, np.tile(0, Alphabet.size-Counts.size))
        I = Alphabet.argsort(kind='mergesort')
        Alphabet = Alphabet[I]
        Counts = Counts[I]
        assert(np.all(Alphabet[1:] != Alphabet[:-1]))
        return Counts, Alphabet
    else:
        assert(Full_Alphabet.shape[0] == 2 and Alphabet.shape[0] == 2)
        Alph1 = np.unique(Full_Alphabet[0, Full_Alphabet[0] != fill_value])
        Alph2 = np.unique(Full_Alphabet[1, Full_Alphabet[1] != fill_value])
        A = np.zeros((2, Alph1.size*Alph2.size), dtype=Full_Alphabet.dtype)
        c = 0
        for j in range(Alph2.size):
            for i in range(Alph1.size):
                A[0, c] = Alph1[i]
                A[1, c] = Alph2[j]
                c = c + 1
        Unseen = np.ones(A.shape[-1], dtype='bool')
        i = j = 0
        while i < Alphabet.shape[-1] and j < A.shape[-1]:
            if np.all(Alphabet[:, i] == A[:, j]):
                Unseen[j] = False
                j = j + 1
                i = i + 1
            elif np.any(Alphabet[:, i] > A[:, j]):
                j = j + 1
            else:
                i = i + 1
        Alphabet = np.hstack((Alphabet, A[:, Unseen]))
        Counts = np.append(Counts, np.tile(0, Alphabet.size-Counts.size))
        # Sort columns
        for i in range(Alphabet.shape[0]):
            I = Alphabet[i].argsort(kind='mergesort')
            Alphabet = Alphabet[:, I]
            Counts = Counts[I]
        assert(np.all(np.any(Alphabet[:, 1:] != Alphabet[:, :-1], axis=0)))
        return Counts, Alphabet


def _autocreate_alphabet(X, fill_value):
    Lengths = np.apply_along_axis(lambda x: np.unique(x).size, axis=-1, arr=X)
    max_length = np.max(Lengths)

    def pad_with_fillvalue(x):
        return np.append(x, np.tile(fill_value, int(max_length-x.size)))
    Alphabet = np.apply_along_axis(lambda x: pad_with_fillvalue(np.unique(x)),
                                   axis=-1, arr=X)
    return (Alphabet, fill_value)


def _cartesian_product_apply(X, Y, function, Alphabet_X=None, Alphabet_Y=None):
    """
    Applies a function to arrays X and Y, each containing discrete random
    variable realisations. (Internal function.)

    **Parameters**:

    X,Y : numpy array (or array-like object such as a list of immutables, as \
    accepted by np.array())
        *cartesian_product==False: X and Y are arrays containing discrete
        random variable realisations, with X.shape==Y.shape. Successive
        realisations of a random variable are indexed by the last axis in the
        respective arrays; multiple random variables in X and Y may be
        specified using preceding axes of the respective arrays (random
        variables are paired **one-to-one** between X and Y). When
        X.ndim==Y.ndim==1, returns a scalar based on calling function(X,Y).
        When X.ndim>1 and Y.ndim>1, returns an array with dimensions
        X.shape[:-1], based on calling function once for each variable pairing
        in the one-to-one relation.

        *cartesian_product==True: X and Y are arrays containing discrete random
        variable realisations, with X.shape[-1]==Y.shape[-1]. Successive
        realisations of a random variable are indexed by the last axis in the
        respective arrays; multiple random variables in X and Y may be
        specified using preceding axes of the respective arrays (random
        variables are paired **many-to-many** between X and Y). When
        X.ndim==Y.ndim==1, returns a scalar based on calling function(X,Y).
        When X.ndim>1 or Y.ndim>1, returns an array with dimensions
        np.append(X.shape[:-1],Y.shape[:-1]), based on calling function once
        for each variable pairing in the many-to-many relation.
    function : function
        A function with two vector-valued arguments.
     """
    assert(X.ndim > 0 and Y.ndim > 0)
    assert(X.size > 0 and Y.size > 0)
    if Alphabet_X is not None or Alphabet_Y is not None:
        assert(Alphabet_X.ndim > 0 and Alphabet_Y.ndim > 0)
        assert(Alphabet_X.size > 0 and Alphabet_Y.size > 0)

    H = np.empty(np.append(X.shape[:-1], Y.shape[:-1]).astype('int'))
    if H.ndim > 0:
        H[:] = np.NaN
    else:
        H = np.float64(np.NaN)

    X = np.reshape(X, (-1, X.shape[-1]))
    Y = np.reshape(Y, (-1, Y.shape[-1]))
    if Alphabet_X is not None or Alphabet_Y is not None:
        Alphabet_X = np.reshape(Alphabet_X, (-1, Alphabet_X.shape[-1]))
        Alphabet_Y = np.reshape(Alphabet_Y, (-1, Alphabet_Y.shape[-1]))
    orig_shape_H = H.shape
    H = np.reshape(H, (-1, 1))

    n = 0
    for i in range(X.shape[0]):
        for j in range(Y.shape[0]):
            if Alphabet_X is not None or Alphabet_Y is not None:
                H[n] = function(X[i], Y[j], Alphabet_X[i], Alphabet_Y[j])
            else:
                H[n] = function(X[i], Y[j])
            n = n + 1

    # Reverse re-shaping
    H = np.reshape(H, orig_shape_H)

    return H


def _determine_number_additional_empty_bins(Counts, Alphabet, Full_Alphabet,
                                            fill_value):
    alphabet_sizes = np.sum(np.atleast_2d(Full_Alphabet) != fill_value,
                            axis=-1)
    if np.any(alphabet_sizes != fill_value):
        joint_alphabet_size = np.prod(alphabet_sizes[alphabet_sizes > 0])
        if joint_alphabet_size <= 0:
            raise ValueError("Numerical overflow detected. Joint alphabet "
                             "size too large.")
    else:
        joint_alphabet_size = 0
    return joint_alphabet_size - \
        np.sum(np.all(np.atleast_2d(Alphabet) != fill_value, axis=0))


def _estimate_probabilities(Counts, estimator, n_additional_empty_bins=0):
    # TODO Documentation should present the following guidelines:
    # 1) Good-Turing may be used if slope requirement satisfied and if
    # unobserved symbols have been defined (TODO Clarify what the requirement
    # is)
    # 2) James-Stein approach may be used as an alternative
    # 3) Dirichlet prior may be used in all other cases

    assert(np.sum(Counts) > 0)
    assert(np.all(Counts.astype('int') == Counts))
    assert(n_additional_empty_bins >= 0)
    Counts = Counts.astype('int')

    if isinstance(estimator, str):
        estimator = estimator.upper().replace(' ', '')

    if np.isreal(estimator) or estimator in ('ML', 'PERKS', 'MINIMAX'):
        if np.isreal(estimator):
            alpha = estimator
        elif estimator == 'PERKS':
            alpha = 1.0 / (Counts.size+n_additional_empty_bins)
        elif estimator == 'MINIMAX':
            alpha = np.sqrt(np.sum(Counts)) / \
                (Counts.size+n_additional_empty_bins)
        else:
            alpha = 0
        Theta = (Counts+alpha) / \
            (1.0*np.sum(Counts) + alpha*(Counts.size+n_additional_empty_bins))
        # Theta_0 is the probability mass assigned to each additional empty bin
        if n_additional_empty_bins > 0:
            Theta_0 = alpha / (1.0*np.sum(Counts) +
                               alpha*(Counts.size+n_additional_empty_bins))
        else:
            Theta_0 = 0

    elif estimator == 'GOOD-TURING':
        # TODO We could also add a Chen-Chao vocabulary size estimator (See
        # Bhat Suma's thesis)

        # The following notation is based on Gale and Sampson (1995)
        # Determine histogram of counts N_r (index r denotes count)
        X = np.sort(Counts)
        B = X[1:] != X[:-1]  # Compute symbol change indicators
        I = np.append(np.where(B), X.size-1)  # Obtain symbol change positions
        N_r = np.zeros(X[I[-1]]+1)
        N_r[X[I]] = np.diff(np.append(-1, I))  # Compute run lengths
        N_r[0] = 0  # Ensures that unobserved symbols do not interfere

        # Compute Z_r, a locally averaged version of N_r
        R = np.where(N_r)[0]
        Q = np.append(0, R[:-1])
        T = np.append(R[1:], 2*R[-1]-Q[-1])
        Z_r = np.zeros_like(N_r)
        Z_r[R] = N_r[R] / (0.5*(T-Q))

        # Fit least squares regression line to plot of log(Z_r) versus log(r)
        x = np.log10(np.arange(1, Z_r.size))
        with np.errstate(invalid='ignore', divide='ignore'):
            y = np.log10(Z_r[1:])
        x = x[np.isfinite(y)]
        y = y[np.isfinite(y)]
        m, c = np.linalg.lstsq(np.vstack([x, np.ones(x.size)]).T, y,
                               rcond=None)[0]
        if m >= -1:
            warnings.warn("Regression slope < -1 requirement in linear "
                          "Good-Turing estimate not satisfied")
        # Compute smoothed value of N_r based on interpolation
        # We need to refer to SmoothedN_{r+1} for all observed values of r
        SmoothedN_r = np.zeros(N_r.size+1)
        SmoothedN_r[1:] = 10**(np.log10(np.arange(1, SmoothedN_r.size)) *
                               m + c)

        # Determine threshold value of r at which to use smoothed values of N_r
        # (SmoothedN_r), as apposed to straightforward N_r.
        # Variance of Turing estimate
        with np.errstate(invalid='ignore', divide='ignore'):
            VARr_T = (np.arange(N_r.size)+1)**2 * \
                (1.0*np.append(N_r[1:], 0)/(N_r**2)) * \
                (1 + np.append(N_r[1:], 0)/N_r)
            x = (np.arange(N_r.size)+1) * 1.0*np.append(N_r[1:], 0) / N_r
            y = (np.arange(N_r.size)+1) * \
                1.0*SmoothedN_r[1:] / (SmoothedN_r[:-1])
            assert(np.isinf(VARr_T[0]) or np.isnan(VARr_T[0]))
            turing_is_sig_diff = np.abs(x-y) > 1.96 * np.sqrt(VARr_T)
        assert(turing_is_sig_diff[0] == np.array(False))
        # NB: 0th element can be safely ignored, since always 0
        T = np.where(turing_is_sig_diff == np.array(False))[0]
        if T.size > 1:
            thresh_r = T[1]
            # Use smoothed estimates from the first non-significant
            # np.abs(SmoothedN_r-N_r) position onwards
            SmoothedN_r[:thresh_r] = N_r[:thresh_r]
        else:
            # Use only non-smoothed estimates (except for SmoothedN_r[-1])
            SmoothedN_r[:-1] = N_r

        # Estimate probability of encountering one particular symbol among the
        # objects observed r times, r>0
        p_r = np.zeros(N_r.size)
        N = np.sum(Counts)
        p_r[1:] = (np.arange(1, N_r.size)+1) * \
            1.0*SmoothedN_r[2:] / (SmoothedN_r[1:-1] * N)
        # Estimate probability of observing any unseen symbol
        p_r[0] = 1.0 * N_r[1] / N

        # Assign probabilities to observed symbols
        Theta = np.array([p_r[r] for r in Counts])
        Theta[Counts == 0] = 0

        # Normalise probabilities for observed symbols, so that they sum to one
        if np.any(Counts == 0) or n_additional_empty_bins > 0:
            Theta = (1-p_r[0]) * Theta / np.sum(Theta)
        else:
            warnings.warn("No unobserved outcomes specified. Disregarding the "
                          "probability mass allocated to any unobserved "
                          "outcomes.")
            Theta = Theta / np.sum(Theta)

        # Divide p_0 among unobserved symbols
        with np.errstate(invalid='ignore', divide='ignore'):
            p_emptybin = p_r[0] / (np.sum(Counts == 0) +
                                   n_additional_empty_bins)
        Theta[Counts == 0] = p_emptybin
        # Theta_0 is the probability mass assigned to each additional empty bin
        if n_additional_empty_bins > 0:
            Theta_0 = p_emptybin
        else:
            Theta_0 = 0

    elif estimator == 'JAMES-STEIN':
        Theta, _ = _estimate_probabilities(Counts, 'ML')
        p_uniform = 1.0 / (Counts.size + n_additional_empty_bins)
        with np.errstate(invalid='ignore', divide='ignore'):
            Lambda = (1-np.sum(Theta**2)) / \
                ((np.sum(Counts)-1) *
                 (np.sum((p_uniform-Theta)**2) +
                  n_additional_empty_bins*p_uniform**2))

        if Lambda > 1:
            Lambda = 1
        elif Lambda < 0:
            Lambda = 0
        elif np.isnan(Lambda):
            Lambda = 1

        Theta = Lambda*p_uniform + (1-Lambda)*Theta
        # Theta_0 is the probability mass assigned to each additional empty bin
        if n_additional_empty_bins > 0:
            Theta_0 = Lambda*p_uniform
        else:
            Theta_0 = 0
    else:
        raise ValueError("invalid value specified for estimator")

    return Theta, Theta_0


def _increment_binary_vector(X):
    carry_1 = False
    x = X[0] ^ True
    if not x:
        carry_1 = True
    X[0] = x
    for i in range(1, X.size):
        x = X[i] ^ carry_1
        if not X[i] and x:
            carry_1 = False
        if X[i] and not x:
            carry_1 = True
        X[i] = x

        if not carry_1:
            break

    return X


def _isnan(X):
    X = np.array(X, copy=False)
    if X.dtype in ('int', 'float'):
        return np.isnan(X)
    else:
        f = np.vectorize(_isnan_element)
        return f(X)


def _isnan_element(x):
    if isinstance(x, type(np.nan)):
        return np.isnan(x)
    else:
        return False


def _map_observations_to_integers(Symbol_matrices, Fill_values):
    assert(len(Symbol_matrices) == len(Fill_values))
    FILL_VALUE = -1
    if np.any([A.dtype != 'int' for A in Symbol_matrices]) or \
            np.any(np.array(Fill_values) != FILL_VALUE):
        L = sklearn.preprocessing.LabelEncoder()
        F = [np.atleast_1d(v) for v in Fill_values]
        L.fit(np.concatenate([A.ravel() for A in Symbol_matrices] + F))
        # TODO make sure to test with various (unusual) data types
        Symbol_matrices = [L.transform(A.ravel()).reshape(A.shape) for A in
                           Symbol_matrices]
        Fill_values = [L.transform(np.atleast_1d(f)) for f in Fill_values]

        for A, f in zip(Symbol_matrices, Fill_values):
            assert(not np.any(A == FILL_VALUE))
            A[A == f] = FILL_VALUE

    assert(np.all([A.dtype == 'int' for A in Symbol_matrices]))
    return Symbol_matrices, FILL_VALUE


def _remove_counts_at_fill_value(Counts, Alphabet, fill_value):
    I = np.any(np.atleast_2d(Alphabet) == fill_value, axis=0)
    if np.any(I):
        Counts = Counts[~I]
        Alphabet = Alphabet.T[~I].T
    return (Counts, Alphabet)


def _sanitise_array_input(X, fill_value=-1):
    # Avoid Python 3 issues with numpy arrays containing None elements
    if np.any(np.equal(X, None)) or fill_value is None:
        X = np.array(X, copy=False)
        assert(np.all(X != NONE_REPLACEMENT))
        M = np.equal(X, None)
        X = np.where(M, NONE_REPLACEMENT, X)
    if fill_value is None:
        X = np.array(X, copy=False)
        fill_value = NONE_REPLACEMENT

    if isinstance(X, (pd.core.frame.DataFrame, pd.core.series.Series)):
        # Create masked array, honouring Dataframe/Series missing entries
        # NB: We transpose for convenience, so that quantities are computed for
        # each column
        X = np.ma.MaskedArray(X, X.isnull())

    if isinstance(X, np.ma.MaskedArray):
        fill_value = X.fill_value

        if np.any(X == fill_value):
            warnings.warn("Masked array contains data equal to fill value")

        if X.dtype.kind in ('S', 'U'):
            kind = X.dtype.kind
            current_dtype_len = int(X.dtype.str.split(kind)[1])
            if current_dtype_len < len(fill_value):
                # Fix numpy's broken array string type behaviour which causes
                # X.filled() placeholder entries to be no longer than
                # non-placeholder entries
                warnings.warn("Changing numpy array dtype internally to "
                              "accommodate fill_value string length")
                M = X.mask
                X = np.array(X.filled(), dtype=kind+str(len(fill_value)))
                X[M] = fill_value
            else:
                X = X.filled()
        else:
            X = X.filled()
    else:
        X = np.array(X, copy=False)

    if X.dtype.kind not in 'biufcmMOSUV':
        raise TypeError("Unsupported array dtype")

    if X.size == 1 and X.ndim == 0:
        X = np.array((X, ))

    return X, np.array(fill_value)


def _verify_alphabet_sufficiently_large(X, Alphabet, fill_value):
    assert(not np.any(X == np.array(None)))
    assert(not np.any(Alphabet == np.array(None)))
    for i in range(X.shape[0]):
        I = X[i] != fill_value
        J = Alphabet[i] != fill_value
        # NB: This causes issues when both arguments contain None. But it is
        # always called after observations have all been mapped to integers.
        if np.setdiff1d(X[i, I], Alphabet[i, J]).size > 0:
            raise ValueError("provided alphabet does not contain all observed "
                             "outcomes")


def _vstack_pad(Arrays, fill_value):
    max_length = max([A.shape[-1] for A in Arrays])
    Arrays = [np.append(A, np.tile(fill_value,
                                   np.append(A.shape[:-1],
                                             max_length -
                                             A.shape[-1]).astype(int)))
              for A in Arrays]
    return np.vstack((Arrays))

# TODO Avoid hack surrounding fill_type None and numpy arrays with Python 3. Remove support for fill_type None?
# TODO Tests for keep_dims
# TODO Should this really be NaN? Is it consistently NaN for all measures?
# drv.entropy([-1,], estimator='PERKS', Alphabet_X = np.arange(100))

# NB: The following tests should also determine what happens when data contain
# None, but fill value is not None
# TODO Test _determine_number_additional_empty_bins using None fill_value etc.
# / add assertions
# TODO Test _append_empty_bins_using_alphabet using None fill_value etc. / add
# assertions
# TODO Test _autocreate_alphabet using None fill_value / add assertions
# TODO Test _remove_counts_at_fill_value / add assertions
# TODO Test _vstack_pad / add assertions
# TODO Run some integration tests using a mixed-type DataFrame
# TODO Run tests using unusual pandas arrangements, such as panels /
# or multi-level Dataframes
