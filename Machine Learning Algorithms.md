# Machine Learning Algorithms

## Probabilistic Supervised Learning

<u>Most supervised learning algorithms in this book are based on estimating a probability distribution p( y | x ). We can do this simply by using maximum likelihood estimation to find the best parameter vector θ for a parametric family of distributions p ( y | x ; θ ) .</u>

### linear regression 

We have already seen that linear regression corresponds to the family
$p ( y | x ; θ ) = N ( y ; θ ^T x , I )$.

### logistic regression 

We can generalize linear regression to the classification scenario by defining a different family of probability distributions. If we have two classes, class 0 and class 1, then we need only specify the probability of one of these classes. The probability of class 1 determines the probability of class 0, because these two values must add up to 1.

The normal distribution over real-valued numbers that we used for linear regression is parametrized in terms of a **mean**. Any value we supply for this mean is valid. A distribution over a binary variable is slightly more complicated, because its mean must always be between 0 and 1. One way to solve this problem is to use the **logistic sigmoid function** to squash the output of the linear function into the interval (0, 1) and interpret that value as a probability:

$p(y=1|x;\theta) = \sigma(\theta^T x)$

$\sigma(x) = \frac {1} {1+exp(-x)}$

This approach is known as logistic regression (a somewhat strange name since we use the model for **classification** rather than regression).

### Maximum Log-Likelyhood and NLL

In the case of linear regression, we were able to find the optimal weights by solving the normal equations. Logistic regression is somewhat more difficult. There is no **closed-form solution** for its optimal weights. Instead, we must search for them by **maximizing the log-likelihood**. We can do this by **minimizing the negative log-likelihood (NLL)** using **gradient descent**.

<u>This same strategy can be applied to essentially any supervised learning problem, by writing down a parametric family of conditional probability distributions over the right kind of input and output variables.</u>

## Support Vector Machines

This model is similar to logistic regression in that it is driven by a linear function $w^T x + b$. Unlike logistic regression, the support vector machine does not provide probabilities, but **only outputs a class identity**. <u>The SVM predicts that the positive class is present when $w ^T x + b$ is positive. Likewise, it predicts that the negative class is present when $w ^T x + b$ is negative.</u>

One key innovation associated with support vector machines is the **kernel trick** . The kernel trick consists of observing that many machine learning algorithms can be written exclusively in terms of dot products between examples. For example, it can be shown that the linear function used by the support vector machine can be
re-written as 

$w^Tx +b = b + \sum_i \alpha_i x^T x^{(i)}$

where $x^{(i)}$ is a training example and **α is a vector of coefficients**. Rewriting the learning algorithm this way allows us to **replace x by the output of a given feature function** $\phi(x)$ and the dot product with a function $k(x , x  ^{(i)}) = \phi(x) · \phi(x^{(i)})$ called a kernel . The · operator represents an inner product analogous to $\phi(x) ^T \phi(x^{(i)})$.

For some feature spaces, we may not use literally the vector inner product. In some infinite dimensional spaces, we need to use other kinds of inner products, for example, inner products based on integration rather than summation. A complete development of these kinds of inner products is beyond the scope of this book.

After replacing dot products with kernel evaluations, we can make predictions using the function

$f(x) = b+\sum_i \alpha_i k(x,x^{(i)})$

This function is nonlinear with respect to x, but the relationship between $\phi(x)$ and f (x) is linear. Also, the relationship between α and f(x) is linear. The kernel-based function is exactly equivalent to preprocessing the data by applying
$\phi( x )$ to all inputs, then learning a linear model in the new transformed space. 

The kernel trick is powerful for two reasons. First, it allows us to learn models that are nonlinear as a function of x using convex optimization techniques that are guaranteed to converge efficiently. This is possible because we consider $\phi$ fixed and optimize only α, i.e., the optimization algorithm can **view the decision function as being linear in a different space.** Second, the kernel function k often admits an implementation that is significantly more computational efficient than naively constructing two $\phi( x )$ vectors and explicitly taking their dot product.

In some cases, $\phi( x )$ can even be infinite dimensional, which would result in an infinite computational cost for the naive, explicit approach. In many cases, k(x , x') is a nonlinear, tractable function of x even when $\phi( x )$  is intractable. As an example of an infinite-dimensional feature space with a tractable kernel, we construct a feature mapping over the non-negative integers x. Suppose that this mapping returns a vector containing x ones followed by infinitely many zeros. We can write a kernel function k(x, x ( i ) ) = min(x, x ( i ) ) that is exactly equivalent to the corresponding infinite-dimensional dot product.

The most commonly used kernel is the Gaussian kernel

$k(u,v) = N(u-v; 0, \sigma^2 I)$

where N (x; μ, Σ) is the standard normal density. This kernel is also known as the **radial basis function (RBF) kernel,** because its value decreases along lines in v space radiating outward from u. The Gaussian kernel corresponds to a dot product in an infinite-dimensional space, but the derivation of this space is less straightforward than in our example of the min kernel over the integers.

We can think of the Gaussian kernel as performing a kind of **template matching** . <u>A training example x associated with training label y becomes a template for class y</u>. When a test point x' is near x according to Euclidean distance, the Gaussian kernel has a large response, indicating that x' is very similar to the x template. The model then puts a large weight on the associated training label y. Overall, the prediction will combine many such training labels weighted by the similarity of the corresponding training examples.

Support vector machines are not the only algorithm that can be enhanced using the kernel trick. Many other linear models can be enhanced in this way. <u>The category of algorithms that employ the kernel trick is known as **kernel machines**</u> or kernel methods ( Williams and Rasmussen , 1996 ; Schölkopf et al., 1999 ).

A major drawback to kernel machines is that <u>the cost of evaluating the decision function is linear in the number of training examples</u>, because the i-th example contributes a term $α_i k(x , x^{(i)} )$ to the decision function. Support vector machines are able to mitigate this by learning an α vector that contains mostly zeros. Classifying a new example then requires evaluating the kernel function only for the training examples that have non-zero $α_i$ . These training examples are known as **support vectors**.

Kernel machines also suffer from a high computational cost of training when the dataset is large. We will revisit this idea in Sec. 5.9 . Kernel machines with generic kernels struggle to generalize well. We will explain why in Sec. 5.11 . **The modern incarnation of deep learning was designed to overcome these limitations of kernel machines.** The current deep learning renaissance began when Hinton et al. ( 2006 ) demonstrated that a neural network could outperform the RBF kernel SVM on the MNIST benchmark.

## Principal Components Analysis

We saw that the principal components analysis algorithm provides a means of **compressing data**. We can also view PCA as an **unsupervised learning algorithm that learns a representation of data**. This representation is based on
two of the criteria for a simple representation described above. **PCA learns a representation that has lower dimensionality than the original input**. It also l**earns a representation whose elements have no linear correlation with each other**. This is a first step toward the criterion of learning representations whose elements are
statistically independent. To achieve full independence, a representation learning algorithm must also remove the nonlinear relationships between variables.

![ml-alg1](/home/wxu/proj2/deep-learning-notes/assets/ml-alg1.png)

PCA learns an orthogonal, linear transformation of the data that projects an input x to a representation z as shown in Fig. 5.8 . In Sec. 2.12 , we saw that **we could learn a one-dimensional representation that best reconstructs the original data** (in the sense of mean squared error) and that this representation actually corresponds to the **first principal component** of the data. Thus we can use **PCA as a simple and effective dimensionality reduction method** that preserves as much of the information in the data as possible (again, as measured by least-squares reconstruction error). In the following, we will study how the PCA representation decorrelates the original data representation X .

Let us consider the m × n -dimensional design matrix X. We will assume that the data has a mean of zero, E[x] = 0. If this is not the case, **the data can easily be centered by subtracting the mean from all examples** in a preprocessing step. The unbiased sample covariance matrix associated with X is given by:

$var[x] = \frac {1} {m-1} X^T X$

PCA finds a representation (through linear transformation) $z = x ^T W$ where Var[ z ] is diagonal.

![ml-alg2](/home/wxu/proj2/deep-learning-notes/assets/ml-alg2.png)

The above analysis shows that when we project the data x to z, via the linear transformation W, the resulting representation has a diagonal covariance matrix (as given by $Σ^2$ ) which immediately implies that the **individual elements of z are mutually uncorrelated.**

This ability of PCA to transform data into a representation where the elements are mutually uncorrelated is a very important property of PCA. It is a simple example of a representation that attempt to disentangle the unknown factors of variation underlying the data. In the case of PCA, this **disentangling** takes the form of finding a **rotation of the input space** (described by W ) that **aligns the principal axes of variance** with the basis of the new representation space associated with z .

While correlation is an important category of dependency between elements of the data, we are also interested in learning representations that disentangle more complicated forms of feature dependencies. For this, we will need more than what can be done with a simple linear transformation.

