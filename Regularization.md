---
typora-root-url: assets
---

# Regularization

A central problem in machine learning is how to make an algorithm that will perform well not just on the training data, but also on new inputs. <u>Many strategies used in machine learning are explicitly designed to reduce the test error, possibly at the expense of increased training error. These strategies are known collectively as regularization</u>. Developing more effective regularization strategies has been one of the major research efforts in the field.

## Parameter Norm Penalties

$\hat J(\theta; X,y) = J(\theta; X, y) + \alpha \Omega(\theta)$

where α ∈ [0, ∞) is a hyperparameter that weights the relative contribution of the norm penalty term, Ω .Setting α to 0 results in no regularization. Larger values of α correspond to more regularization.

we typically choose to use a parameter norm penalty Ω that **penalizes only the weights (w)** of the affine transformation at each layer and **leaves the biases unregularized**.

It is reasonable to use the same weight decay at all layers just to reduce the size of search space.

### $L^2$ Parameter Regularization

The $L^ 2$ parameter norm penalty commonly known as **weight decay**.

$\Omega(\theta) = \frac {1} {2} \|w\|^2_2$

### $L^1$ Regularization

$\Omega(\theta) = |w\|_1 = \sum_i |w_i|$

## Norm Penalties as Constrained Optimization

Sometimes we wish not only to maximize or minimize a function f(x) over **all** possible values of x. Instead we may wish to find the maximal or minimal value of f (x) for values of **x in some set S.** This is known as **constrained optimization**. Points x that lie within the set S are called **feasible points** in constrained optimization terminology.

For example, if we want to minimize f(x) for $x \in R^2$ with x constrained to have exactly unit $L^2$ norm, we can instead minimize $g(θ ) = f ([cos θ, sin θ] ^T )$with respect to θ, then return [cos θ, sin θ] as the solution to the original problem.

The **Karush–Kuhn–Tucker (KKT) approach** provides a very general solution to constrained optimization. With the KKT approach, we introduce a new function called the **generalized Lagrangian** or **generalized Lagrange function** .

To define the Lagrangian, we first need to <u>describe S in terms of equations and inequalities</u>. We want a description of S in terms of m functions $g^{(i)}$ and n functions $h^{(j)}$ so that $S= \{\forall i, g^{(i)}(x) = 0 and \forall j, h^{(j)}(x)\leq 0 \}$ 

We introduce new variables λ i and α j for each constraint, these are called the KKT multipliers. The generalized Lagrangian is then defined as

$L(x,\lambda, \alpha) = f(x) + \sum_i \lambda_ig^{(i)}(x) + \sum_j \alpha_j h^{(j)}(x)$

We can now solve a constrained minimization problem using unconstrained optimization of the generalized Lagrangian.

## Regularization and Under-Determined Problems

In some cases, regularization is necessary for machine learning problems to be properly defined. Many linear models in machine learning, including linear regression and PCA, depend on inverting the matrix $ X^T X$. This is not possible whenever $X^T X$ is singular. This matrix can be singular whenever the <u>data generating distribution truly has no variance in some direction</u>, or when <u>no variance is observed</u> <u>i</u>n some direction because there are fewer examples (rows of X ) than input features (columns of X ). In this case, many forms of regularization correspond to inverting
$X ^T X + α I$ instead. This regularized matrix is guaranteed to be invertible.

## Dataset Augmentation

The best way to make a machine learning model generalize better is to train it on more data. Of course, in practice, the amount of data we have is limited. One way to get around this problem is to **create fake data and add it to the training set.** For some machine learning tasks, it is reasonably straightforward to create new fake data.

This approach is easiest for classification. A classifier needs to take a complicated, high dimensional input x and summarize it with a single category identity y. This means that the main task facing a classifier is to be invariant to a wide variety of transformations. We can generate new (x, y) pairs easily just by transforming the x inputs in our training set.

Dataset augmentation has been a particularly effective technique for a specific classification problem: object recognition. Images are high dimensional and include an enormous variety of factors of variation, many of which can be easily simulated. Operations like <u>translating the training images a few pixels in each direction can</u> <u>often greatly improve generalization, even if the model has already been designed to</u> be partially translation invariant by using the convolution and pooling techniques described in Chapter 9 . Many other operations such as <u>rotating the image or</u>
<u>scaling the image</u> have also proven quite effective.

Dataset augmentation is effective for **speech recognition** tasks as well.

Injecting noise in the input to a neural network (Sietsma and Dow , 1991 ) can also be seen as a form of data augmentation. When comparing machine learning benchmark results, it is important to take the effect of dataset augmentation into account.

## Noise Robustness

## Parameter Sharing

More popular way is to use **constraints**: to <u>force sets of parameters to be equal.</u> This method of regularization is often referred to as **parameter sharing**, where we interpret the various models or model components as sharing a unique set of parameters. A significant advantage of parameter sharing over regularizing the parameters to be close (via a norm penalty) is that only a subset of the parameters (the unique set) need to be stored in memory. In certain
models—such as the convolutional neural network—this can lead to significant reduction in the memory footprint of the model.

Convolutional Neural Networks By far the most popular and extensive use of parameter sharing occurs in convolutional neural networks (CNNs) applied to computer vision. <u>Natural images have many statistical properties that are invariant to translation.</u> For example, a photo of a cat remains a photo of a cat if it is translated one pixel
to the right. CNNs take this property into account by sharing parameters across multiple image locations. The same feature (a hidden unit with the same weights) is computed over different locations in the input. This means that we can find a cat with the same cat detector whether the cat appears at column i or column i + 1 in the image.

Parameter sharing has allowed CNNs to dramatically lower the number of unique model parameters and to significantly increase network sizes without requiring a corresponding increase in training data. It remains one of the best examples of how to effectively incorporate domain knowledge into the network architecture.

## Sparse Representations

Weight decay acts by placing a penalty directly on the model parameters. Another strategy is to place a penalty on the activations of the units in a neural network, encouraging their activations to be sparse. This indirectly imposes a complicated penalty on the model parameters.

## Bagging and Other Ensemble Methods

**Bagging** (short for **bootstrap aggregating** ) is a <u>technique for reducing generalization error by combining several models</u> ( Breiman , 1994 ). <u>The idea is to train several different models separately, then have all of the models vote on the output for test examples.</u> This is an example of a general strategy in machine learning called **model averaging**. Techniques employing this strategy are known as **ensemble methods**.

<u>The reason that model averaging works is that different models will usually not make all the same errors on the test set</u>.

Consider for example a set of k regression models. Suppose that each model makes an error  $\epsilon_i$ on each example, with the errors drawn from a zero-mean multivariate normal distribution with variances $E [\epsilon^ 2_i ] = v$ and covariances $E [\epsilon_ i \epsilon_ j ] = c$. Then the error made by the average prediction of all the ensemble models is $\frac {1} {k} \sum_i \epsilon_i$  . The expected squared error of the ensemble predictor is

$E[(\frac {1} {k} \sum_i \epsilon_i)^2] = \frac {1} {k^2}E[\sum_i (\epsilon_i^2 + \sum_{j:j\neq i}\epsilon_j \epsilon_j)] = \frac {1} {k} v + \frac {k-1} {k}c$

<u>In the case where the errors are perfectly correlated and c = v, the mean squared error reduces to v, so the model averaging does not help at all. In the case where the errors are perfectly uncorrelated and c = 0, the expected squared error of the ensemble is only</u> $\frac {1} {k} v$. **This means that the expected squared error of the ensemble**
**decreases linearly with the ensemble size**. <u>In other words, on average, the ensemble will perform at least as well as any of its members, and if the members make independent errors, the ensemble will perform significantly better than its members.</u>

<u>Different ensemble methods construct the ensemble of models in different ways.</u> For example, each member of the ensemble could be formed by training a completely different kind of model using a different algorithm or objective function. <u>**Bagging** is a method that allows the same kind of model, training algorithm and objective function to be reused several times.</u>

<u>Specifically, bagging involves constructing k different datasets. Each dataset has the same number of examples as the original dataset, but each dataset is constructed by sampling with replacement from the original dataset. This means  that, with high probability, each dataset is missing some of the examples from the original dataset and also contains several duplicate examples</u> (on average around 2/3 of the examples from the original dataset are found in the resulting training set, if it has the same size as the original). <u>Model i is then trained on dataset i</u>. The differences between which examples are included in each dataset result in differences between the trained models. See Fig. 7.5 for an example. <u>Neural networks reach a wide enough variety of solution points that they can often benefit from model averaging even if all of the models are trained on the same dataset. Differences in random initialization, random selection of minibatches, differences in hyperparameters, or different outcomes of non-deterministic imple-</u>
<u>mentations of neural networks are often enough to cause different members of the ensemble to make partially independent errors.</u>

![bagging](bagging.png)

Model averaging is an extremely powerful and reliable method for reducing generalization error. Its use is usually discouraged when benchmarking algorithms for scientific papers, because **any machine learning algorithm can benefit substantially from model averaging** at the price of increased computation and memory. For this reason, benchmark comparisons are usually made using a single model. <u>Machine learning contests are usually won by methods using model averaging over dozens of models</u>. A recent prominent example is the Netflix Grand
Prize (Koren , 2009 ).

Not all techniques for constructing ensembles are designed to make the ensemble **more regularized** than the individual models. For example, a technique called **boosting** (Freund and Schapire , 1996b , a ) constructs an ensemble with **higher capacity** than the individual models. Boosting has been applied to build ensembles of neural networks (Schwenk and Bengio , 1998 ) by incrementally adding neural networks to the ensemble. Boosting has also been applied interpreting an individual neural network as an ensemble ( Bengio et al. , 2006a ), incrementally adding hidden units to the neural network.

## Dropout

Dropout (Srivastava et al., 2014 ) provides a computationally inexpensive but powerful method of regularizing a broad family of models. To a first approximation, dropout can be thought of as a <u>method of making bagging practical for ensembles of very many large neural networks</u>. Bagging involves training multiple models, and evaluating multiple models on each test example. This seems impractical when each model is a large neural network, since training and evaluating such networks is costly in terms of runtime and memory. <u>It is common to use ensembles of five to ten neural networks— Szegedy et al. ( 2014a ) used six to win the ILSVRC— but more than this rapidly becomes unwieldy</u>. <u>Dropout provides an inexpensive approximation to training and evaluating a bagged ensemble of **exponentially many** neural networks.</u>

Specifically, dropout trains the ensemble consisting of all sub-networks that can be formed by removing non-output units from an underlying base network, as illustrated in Fig. 7.6 . In most modern neural networks, based on a series of **affine transformations and nonlinearities**, we can **effectively remove a unit from a network by multiplying its output value by zero**. This procedure requires some slight modification for models such as **radial basis function** networks, which take the difference between the unit’s state and some reference value. Here, we present the dropout algorithm in terms of multiplication by zero for simplicity, but it can be trivially modified to work with other  operations that remove a unit from the network.

![dropout2](dropout.png)

Recall that to learn with bagging, we define k different models, construct k different datasets by sampling from the training set with replacement, and then train model i on dataset i. <u>Dropout aims to approximate this process, but with an exponentially large number of neural networks</u>. <u>Specifically, to train with dropout, we use a minibatch-based learning algorithm that makes small steps, such as stochastic gradient descent. Each time we load an example into a minibatch, **we randomly sample a different binary mask to apply to all of the input and hidden units** in the network. The mask for each unit is sampled independently from all of the others. The probability of sampling a mask value of one (causing a unit to be included) is a hyperparameter fixed before training begins. It is not a function of the current value of the model parameters or the input example. Typically, an input unit is included with probability 0.8 and a hidden unit is included with probability 0.5. We then run forward propagation, back-propagation, and the learning update</u> as usual. Fig. 7.7 illustrates how to run forward propagation with dropout.

More formally, suppose that a mask vector μ specifies which units to include, and J(θ , μ ) defines the cost of the model defined by parameters θ and mask μ. Then dropout training consists in minimizing $E_μ J( θ , μ )$. The expectation contains exponentially many terms but we can obtain an unbiased estimate of its gradient by sampling values of μ .

Dropout training is not quite the same as bagging training. In the case of bagging, the models are all independent. In the case of dropout, the models share parameters, with each model inheriting a different subset of parameters from the parent neural network. This parameter sharing makes it possible to represent an exponential number of models with a tractable amount of memory. In the case of bagging, each model is trained to convergence on its respective training set. In the case of dropout, typically most models are not explicitly trained at all—usually, the model is large enough that it would be infeasible to sample all possible sub-networks within the lifetime of the universe. Instead, a tiny fraction of the possible sub-networks are each trained for a single step, and the parameter sharing causes the remaining sub-networks to arrive at good settings of the parameters. These are the only differences. Beyond these, dropout follows the bagging algorithm. For example, the training set encountered by each sub-network is indeed a subset of the original training set sampled with replacement.

![dropout2](dropout2.png)

**To make a prediction, a bagged ensemble must accumulate votes from all of its members. We refer to this process as inference in this context**. So far, our description of bagging and dropout has not required that the model be explicitly probabilistic. Now, we assume that the model’s role is to output a probability distribution. In the case of bagging, each model i produces a probability distribution $p^{(i)} (y | x)$. The prediction of the ensemble is given by the arithmetic mean of all of these distributions,

$\frac {1} {k} \sum_{i=1}^k p^{(i)}(y|x)$

In the case of dropout, each sub-model defined by mask vector μ defines a probability distribution p(y | x , μ). The arithmetic mean over all masks is given by

$\sum_{\mu} p(\mu)p(y|x,\mu)$

where p(μ) is the probability distribution that was used to sample μ at training time.

Because this sum includes an exponential number of terms, it is intractable to evaluate except in cases where the structure of the model permits some form of simplification. So far, deep neural nets are not known to permit any tractable simplification. Instead, we can approximate the inference with sampling, by averaging together the output from many masks. **Even 10-20 masks are often sufficient to obtain good performance.**

However, there is an even better approach, that allows us to obtain a good approximation to the predictions of the entire ensemble, at the cost of only one forward propagation. To do so, we change to using the **geometric mean** rather than the arithmetic mean of the ensemble members’ predicted distributions. Warde-Farley et al. ( 2014 ) present arguments and empirical evidence that the geometric mean performs comparably to the arithmetic mean in this context.

The geometric mean of multiple probability distributions is not guaranteed to be a probability distribution. To guarantee that the result is a probability distribution, we impose the requirement that none of the sub-models assigns probability 0 to any event, and we renormalize the resulting distribution. The unnormalized probability
distribution defined directly by the geometric mean is given by

$\hat p_{ensemble}(y|x) = \sqrt[2^d] {\prod_{\mu} {p(y|x,\mu)}}$

where d is the number of units that may be dropped. Here we use a uniform distribution over μ to simplify the presentation, but non-uniform distributions are also possible. To make predictions we must re-normalize the ensemble:

$p_{emsemble} (y|x) = \frac {\hat p_{ensemble}(y|x)} {\sum_{y'} \hat p_{ensemble} (y'|x)}$

**A key insight ( Hinton et al. , 2012c ) involved in dropout is that we can approximate p ensemble by evaluating p(y | x) in one model: the model with all units, but with the weights going out of unit i multiplied by the probability of including unit i.** The motivation for this modification is to <u>capture the right expected value of the output from that unit</u>. We call this approach the **weight scaling inference rule**. There is <u>not yet any theoretical argument</u> for the accuracy of this approximate inference rule in deep nonlinear networks, but empirically it performs very well.

Because we usually use an inclusion probability of 1/2 , the weight scaling rule usually amounts to dividing the weights by 2 at the end of training, and then using the model as usual. Another way to achieve the same result is to multiply the states of the units by 2 during training. Either way, the goal is to make sure that the expected total input to a unit at test time is roughly the same as the expected total input to that unit at train time, even though half the units at train time are missing on average.

![dropout2](dropout3.png)

![dropout2](dropout4.png)



<u>The weight scaling rule is also exact in other settings, including regression networks with conditionally normal outputs, and deep networks that have hidden layers without nonlinearities. However, the weight scaling rule is only an approximation for deep models that have nonlinearities.</u> Though the approximation has not been theoretically characterized, it often works well, empirically. Goodfellow et al. ( 2013a ) found experimentally that the weight scaling approximation can work better (in terms of classification accuracy) than Monte Carlo approximations to the ensemble predictor. This held true even when the Monte Carlo approximation was allowed to sample up to 1,000 sub-networks. Gal and Ghahramani ( 2015 ) found that some models obtain better classification accuracy using twenty samples and the Monte Carlo approximation. It appears that the optimal choice of inference approximation is problem-dependent.

Srivastava et al. ( 2014 ) showed that **dropout is more effective** than other standard computationally inexpensive regularizers, such as **weight decay, filter norm constraints and sparse activity regularization**. Dropout may also be combined with other forms of regularization to yield a further improvement.

One advantage of dropout is that it is very computationally cheap. Using dropout during training requires only O(n) computation per example per update, to generate n random binary numbers and multiply them by the state.  Depending on the implementation, it may also require O (n) memory to store these binary numbers until the back-propagation stage. Running inference in the trained model has the same cost per-example as if dropout were not used, though we must pay the cost of dividing the weights by 2 once before beginning to run inference on examples.

Another significant advantage of dropout is that it does not significantly limit the type of model or training procedure that can be used. It **works well with nearly any model** that uses a **distributed representation** and can be trained with **stochastic gradient descent**. This includes feedforward neural networks, probabilistic models such as restricted Boltzmann machines (Srivastava et al., 2014 ), and recurrent neural networks (Bayer and Osendorfer , 2014 ; Pascanu et al., 2014a ). Many other regularization strategies of comparable power impose more severe restrictions on the architecture of the model.

Though the cost per-step of applying dropout to a specific model is negligible, the cost of using dropout in a complete system can be significant. Because dropout is a regularization technique, **it reduces the effective capacity of a model**. <u>To offset this effect, we must increase the size of the model</u>. Typically the optimal validation set error is much lower when using dropout, but this comes at the cost of a much larger model and many more iterations of the training algorithm. For very large datasets, regularization confers little reduction in generalization error. In these
cases, the computational cost of using dropout and larger models may outweigh the benefit of regularization.

When **extremely few labeled training examples** are available, dropout is less effective. Bayesian neural networks ( Neal , 1996 ) outperform dropout on the Alternative Splicing Dataset ( Xiong et al. , 2011 ) where fewer than 5,000 examples are available (Srivastava et al., 2014 ). When additional unlabeled data is available, unsupervised feature learning can gain an advantage over dropout. Wager et al. ( 2013 ) showed that, when applied to linear regression, dropout is equivalent to $L^2$ weight decay, with a different weight decay coefficient for each input feature. The magnitude of each feature’s weight decay coefficient is determined by its variance. Similar results hold for other linear models. For deep models, dropout is not equivalent to weight decay.

The stochasticity used while training with dropout is not necessary for the approach’s success. It is just a means of approximating the sum over all sub-models. Wang and Manning ( 2013 ) derived analytical approximations to this
marginalization. Their approximation, known as fast dropout resulted in faster convergence time due to the reduced stochasticity in the computation of the gradient. This method can also be applied at test time, as a more principled
(but also more computationally expensive) approximation to the average over all sub-networks than the weight scaling approximation. Fast dropout has been used to nearly match the performance of standard dropout on small neural network problems, but has not yet yielded a significant improvement or been applied to a large problem.

Just as stochasticity is not necessary to achieve the regularizing effect of dropout, it is also not sufficient. To demonstrate this, Warde-Farley et al. ( 2014 ) designed control experiments using a method called dropout boosting that they designed to use exactly the same mask noise as traditional dropout but lack its regularizing effect. Dropout boosting trains the entire ensemble to jointly maximize the log-likelihood on the training set. In the same sense that traditional dropout is analogous to bagging, this approach is analogous to boosting. As intended, experiments with dropout boosting show almost no regularization effect compared to training the entire network as a single model. This demonstrates that the interpretation of dropout as bagging has value beyond the interpretation of dropout as robustness to noise. The regularization effect of the bagged ensemble is only achieved when the stochastically sampled ensemble members are trained to perform well independently of each other.

Dropout has inspired other stochastic approaches to training exponentially large ensembles of models that share weights. DropConnect is a special case of dropout where each product between a single scalar weight and a single hidden unit state is considered a unit that can be dropped (Wan et al., 2013 ). Stochastic pooling is a form of randomized pooling (see Sec. 9.3 ) for building ensembles of convolutional networks with each convolutional network attending to different spatial locations of each feature map. So far, dropout remains the most widely used implicit ensemble method.

One of the key insights of dropout is that training a network with stochastic behavior and making predictions by averaging over multiple stochastic decisions implements a form of bagging with parameter sharing. Earlier, we described dropout as bagging an ensemble of models formed by including or excluding units. However, there is no need for this model averaging strategy to be based on inclusion and exclusion. In principle, any kind of random modification is admissible. In practice, we must choose modification families that neural networks are able
to learn to resist. Ideally, we should also use model families that allow a fast approximate inference rule. We can think of any form of modification parametrized by a vector μ as training an ensemble consisting of p(y | x , μ) for all possible values of μ. There is no requirement that μ have a finite number of values. For example, μ can be real-valued. Srivastava et al. ( 2014 ) showed that multiplying the weights by μ ∼ N (1, I) can outperform dropout based on binary masks. Because E [μ] = 1 the standard network automatically implements approximate inference in the ensemble, without needing any weight scaling.

So far we have described dropout purely as a means of performing efficient, approximate bagging. However, there is another view of dropout that goes further than this. Dropout trains not just a bagged ensemble of models, but an **ensemble of models that share hidden units**. This means each hidden unit must be able to perform well regardless of which other hidden units are in the model. Hidden units must be prepared to be swapped and interchanged between models. Hinton et al. ( 2012c ) were inspired by an idea from biology: **sexual reproduction**, which involves swapping genes between two different organisms, creates evolutionary pressure for genes to become not just good, but to become readily swapped between different organisms. Such genes and such features are very robust to changes in their environment because they are not able to incorrectly adapt to unusual features of any one organism or model. **Dropout thus regularizes each hidden unit to be not merely a good feature but a feature that is good in many contexts.** Warde-Farley et al. ( 2014 ) compared dropout training to training of large ensembles and concluded that dropout offers additional improvements to generalization error beyond those obtained by ensembles of independent models.

It is important to understand that a large portion of the power of dropout arises from the fact that **the masking noise is applied to the hidden units**. This can be seen as **a form of highly intelligent, adaptive destruction of the information content of the input** rather than **destruction of the raw values of the input**. <u>For example, if the model learns a hidden unit $h_i$ that detects a face by finding the nose, then dropping $h_i$ corresponds to erasing the information that there is a nose in the image. The model must learn another $h_i$ , either that redundantly encodes the</u>
<u>presence of a nose, or that detects the face by another feature, such as the mouth.</u> Traditional noise injection techniques that add unstructured noise at the input are not able to randomly erase the information about a nthe image is removed. <u>Destroying extracted features rather than original values allows the destruction process to make use of all of the knowledge about the input distribution that the model has acquired so far.</u>

Another important aspect of dropout is that the noise is **multiplicative**. If the noise were additive with fixed scale, then a rectified linear hidden unit $h_i$ with added noise  could simply learn to have $h_i$ become very large in order to make the added noise  insignificant by comparison. Multiplicative noise does not allow such a pathological solution to the noise robustness problem. 

Another deep learning algorithm, **batch normalization**, reparametrizes the model in a way that introduces both additive and multiplicative noise on the hidden units at training time. The primary purpose of batch normalization is to improve optimization, but the noise can have a regularizing effect, and sometimes makes dropout unnecessary. Batch normalization is described further in Sec. 8.7.1 .ose from an image of a face unless the magnitude of the noise is so great that nearly all of the information in 

## Adversarial Training

## Tangent Distance, Tangent Prop, and Manifold Tangent Classifier

