## Part I ##

From *Elements of Statistical Learning by Tibshirani*

Consider the two possible scenarios:
- **Scenario 1**: The training data in each class were generated from bivariate Gaussian distributions with uncorrelated components and different means.
- **Scenario 2**: The training data in each class came from a mixture of 10 low-variance Gaussian distributions, with individual means themselves distributed as Gaussian.

A mixture of Gaussians is best described in terms of the generative model. One first generates a discrete variable that determines which of 14 2. Overview of Supervised Learning the component Gaussians to use, and then generates an observation from the chosen density.

In particular **linear regression** is more appropriate for Scenario 1 above, while **nearest neighbors** are more suitable for Scenario 2.

Referring to [Can someone please explain to me what the particular scenarios mean?](http://stats.stackexchange.com/questions/81197/can-someone-please-explain-to-me-what-the-particular-scenarios-mean)

## Part II ##

From *Elements of Statistical Learning by Tibshirani*

So both k-nearest neighbors and least squares end up approximating conditional expectations by averages. But they differ dramatically in terms of model assumptions:
-  Least squares assumes f(x) is well approximated by a globally linear function.
-  k-nearest neighbors assumes f(x) is well approximated by a locally constant function.