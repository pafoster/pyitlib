pyitlib is an MIT-licensed library of information-theoretic methods for data analysis and machine learning, implemented in Python and NumPy.

API documentation is available online at https://pafoster.github.io/pyitlib/.

pyitlib implements the following 19 measures on discrete random variables:

* Entropy
* Joint entropy
* Conditional entropy
* Cross entropy
* Kullback-Leibler divergence
* Symmetrised Kullback-Leibler divergence
* Jensen-Shannon divergence
* Mutual information
* Normalised mutual information (7 variants)
* Variation of information
* Lautum information
* Conditional mutual information
* Co-information
* Interaction information
* Multi-information
* Binding information
* Residual entropy
* Exogenous local information
* Enigmatic information

The following estimators are available for each of the measures:

* Maximum likelihood
* Maximum a posteriori
* James-Stein
* Good-Turing

Missing data are supported, either using placeholder values or NumPy masked arrays.

Installation and codebase
-------------------------
pyitlib is listed on the Python Package Index at https://pypi.python.org/pypi/pyitlib/ and may be installed using ``pip`` as follows:

.. code:: python

    pip install pyitlib

The codebase for pyitlib is available at https://github.com/pafoster/pyitlib.


Notes for getting started
-------------------------

Import the module ``discrete_random_variable``, as well as NumPy:

.. code:: python

    import numpy as np
    from pyitlib import discrete_random_variable as drv

The respective methods implemented in ``discrete_random_variable`` accept NumPy arrays as input. Let's compute the entropy for an array containing discrete random variable realisations, based on maximum likelihood estimation and quantifying entropy in bits:

.. code:: python

    >>> X = np.array((1,2,1,2))
    >>> drv.entropy(X)
    array(1.0)

NumPy arrays are created automatically for any input which isn't of the required type, by passing the input to np.array(). Let's compute entropy, again based on maximum likelihood estimation, but this time using list input and quantifying entropy in nats:

.. code:: python

    >>> drv.entropy(['a', 'b', 'a', 'b'], base=np.exp(1))
    array(0.6931471805599453)

Those methods with the suffix ``_pmf`` operate on arrays specifying probability mass assignments. For example, the analogous method call for computing the entropy of the preceding random variable realisations (with estimated equi-probable outcomes) is:

.. code:: python

    >>> drv.entropy_pmf([0.5, 0.5], base=np.exp(1))
    0.69314718055994529

It's possible to specify missing data using placeholder values (the default placeholder value is ``-1``). Elements equal to the placeholder value are subsequently ignored:

.. code:: python

    >>> drv.entropy([1, 2, 1, 2, -1])
    array(1.0)

In measures expressible in terms of joint entropy (such as conditional entropy, mutual information etc.), equally many realisations of respective random variables are required (with realisations coupled using a common index). Any missing data for random variable ``X`` results in the corresponding realisations for random variable ``Y`` being ignored, and vice versa. Thus, the following method calls yield equivalent results (note use of alternative placeholder value ``None``):

.. code:: python

    >>> drv.entropy_conditional([1,2,2,2], [1,1,2,2])
    array(0.5)
    >>> drv.entropy_conditional([1,2,2,2,1], [1,1,2,2,None], fill_value=None)
    array(0.5)

It's alternatively possible to specify missing data using NumPy masked arrays:

.. code:: python

    >>> Z = np.ma.array((1,2,1), mask=(0,0,1))
    >>> drv.entropy(Z)
    array(1.0)

In combination with any estimator other than maximum likelihood, it may be useful to specify alphabets containing unobserved outcomes. For example, we might seek to estimate the entropy in bits for the sequence of realisations ``[1,1,1,1]``. Using maximum a posteriori estimation combined with the Perks prior (i.e. pseudo-counts of 1/L for each of L possible outcomes) and based on an alphabet specifying L=100 possible outcomes, we may use:

.. code:: python

    >>> drv.entropy([1,1,1,1], estimator='PERKS', Alphabet_X = np.arange(100))
    array(2.030522626645241)

Multi-dimensional array input is supported based on the convention that *leading dimensions index random variables, with the trailing dimension indexing random variable realisations*. Thus, the following array specifies realisations for 3 random variables:

.. code:: python

    >>> X = np.array(((1,1,1,1), (1,1,2,2), (1,1,2,2)))
    >>> X.shape
    (3, 4)

When using multi-dimensional arrays, any alphabets must be specified separately for each random variable represented in the multi-dimensional array, using placeholder values (or NumPy masked arrays) to pad out any unequally sized alphabets:

.. code:: python

    >>> drv.entropy(X, estimator='PERKS', Alphabet_X = np.tile(np.arange(100),(3,1))) # 3 alphabets required
    array([ 2.03052263,  2.81433872,  2.81433872])

    >>> A = np.array(((1,2,-1), (1,2,-1), (1,2,3))) # padding required
    >>> drv.entropy(X, estimator='PERKS', Alphabet_X = A)
    array([ 0.46899559,  1.        ,  1.28669267])

For ease of use, those methods operating on two random variable array arguments (such as ``entropy_conditional``, ``information_mutual`` etc.) may be invoked with a single multi-dimensional array. In this way, we may compute mutual information for all pairs of random variables represented in the array as follows:

.. code:: python

    >>> drv.information_mutual(X)
    array([[ 0.,  0.,  0.],
           [ 0.,  1.,  1.],
           [ 0.,  1.,  1.]])

The above is equivalent to setting the ``cartesian_product`` parameter to ``True`` and specifying two random variable array arguments explicitly:

.. code:: python

    >>> drv.information_mutual(X, X, cartesian_product=True)
    array([[ 0.,  0.,  0.],
           [ 0.,  1.,  1.],
           [ 0.,  1.,  1.]])

By default, those methods operating on several random variable array arguments don't determine all combinations of random variables exhaustively. Instead a one-to-one mapping is performed:

.. code:: python

    >>> drv.information_mutual(X, X) # Mutual information between 3 pairs of random variables
    array([ 0.,  1.,  1.])

    >>> drv.entropy(X) # Mutual information equivalent to entropy in above case
    array([ 0.,  1.,  1.])

pyitlib provides basic support for pandas DataFrames/Series. Both these types are converted to NumPy masked arrays internally, while masking those data recorded as missing (based on .isnull()). Note that due to indexing random variable realisations using the trailing dimension of multi-dimensional arrays, we typically need to transpose DataFrames when estimating information-theoretic quantities:

.. code:: python

    >>> import pandas
    >>> df = pandas.read_csv('https://raw.githubusercontent.com/veekun/pokedex/master/pokedex/data/csv/pokemon.csv')
    >>> df = df[['height', 'weight', 'base_experience']].apply(lambda s: pandas.qcut(s, 10, labels=False)) # Bin the data
    >>> drv.information_mutual_normalised(df.T) # Transposition required for comparing columns
    array([[ 1.        ,  0.32472696,  0.17745753],
           [ 0.32729034,  1.        ,  0.13343504],
           [ 0.17848175,  0.13315407,  1.        ]])
