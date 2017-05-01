# Optimization

This chapter focuses on one particular case of optimization: **finding the parameters θ of a neural network that significantly reduce a cost function J(θ), which typically includes a performance measure evaluated on the entire training set as well as additional regularization terms.**

Typically, the cost function can be written as an average over the training set,such as

$J(\theta) = E_{x,y\sim\hat p_{data}} L(f(x;\theta), y)$

where L is the per-example loss function, f (x; θ) is the predicted output when the input is x, $\hat p_{data}$ is the empirical distribution. In the supervised learning case, y is the target output.

The goal of a machine learning algorithm is to reduce the expected generalization error given by

$J^*(\theta) = E_{x,y\sim p_{data}} L(f(x;\theta), y)$

This quantity is known as the risk . We emphasize here that the expectation is taken over the true underlying distribution $p_{data}$ . If we knew the true distribution $p_{data} (x, y)$, risk minimization would be an optimization task solvable by an optimization algorithm. However, when we do not know $p_{data} (x, y)$ but only have a training set of samples, we have a machine learning problem. The training process based on minimizing this average training error is known as **empirical risk minimization**.

Carefully designing the objective function and constraints to ensure that the optimization problem is **convex**.

## Challenges

### Hessian ill-conditioning

Ill-conditioning can manifest by causing SGD to get “stuck” in the sense that even very small steps increase the cost function.

### Local Minima

With non-convex functions, such as neural nets, it is possible to have many local minima.

## Basic Algorithms

### Stochastic Gradient Descent

![sgd](/home/wxu/proj2/deep-learning-notes/assets/sgd.png)

A crucial parameter for the SGD algorithm is the learning rate. <u>In practice, it is necessary to gradually decrease the learning rate over time.</u>

### Momentum

The method of momentum (Polyak , 1964 ) is designed to accelerate learning, especially in the face of high curvature, small but consistent gradients, or noisy gradients.

![sgd-momentum](/home/wxu/proj2/deep-learning-notes/assets/sgd-momentum.png)

![sgd-momentum2](/home/wxu/proj2/deep-learning-notes/assets/sgd-momentum2.png)

### Nesterov Momentum

Thus one can interpret Nesterov momentum as attempting to add a correction factor to the standard method of momentum.

![sgd-momentum3](/home/wxu/proj2/deep-learning-notes/assets/sgd-momentum3.png)

![sgd-momentum4](/home/wxu/proj2/deep-learning-notes/assets/sgd-momentum4.png)

## Optimization Strategies and Meta-Algorithms

### Batch Normalization

Batch normalization ( Ioffe and Szegedy , 2015 ) is one of the most exciting recent innovations.

Batch normalization provides an elegant way of reparametrizing almost any deep network. <u>The reparametrization significantly reduces the problem of coordinating updates across many layers.</u> Batch normalization can be applied to any input or hidden layer in a network. Let H be a minibatch of activations of the layer to normalize, arranged as a design matrix, with the activations for each example appearing in a row of the matrix. To normalize H , we replace it with

$H' = \frac {H-\mu} {\sigma}$

where μ is a vector containing the mean of each unit and σ is a vector containing the standard deviation of each unit.

### Coordinate Descent

