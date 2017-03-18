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

Missing data are supported, either using fill values or NumPy masked arrays.
