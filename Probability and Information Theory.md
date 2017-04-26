---
typora-root-url: assets
---

# Probability and Information Theory

### Uncertainty

**Probability** theory is a mathematical framework for representing uncertain statements. It provides a means of quantifying uncertainty and axioms for deriving new uncertain statements. In artificial intelligence applications, we use probability
theory in two major ways. First, the laws of probability tell us how AI systems should reason, so we design our algorithms to compute or approximate various expressions derived using probability theory. Second, we can use probability and statistics to theoretically analyze the behavior of proposed AI systems.

While probability theory allows us to make uncertain statements and reason in the presence of uncertainty, **information** allows us to <u>quantify the amount of uncertainty</u> in a probability distribution.

There are three possible sources of uncertainty:

1. **Inherent stochasticity in the system being modeled**. For example, most interpretations of <u>quantum mechanics</u> describe the dynamics of subatomic particles as being probabilistic. We can also create theoretical scenarios that we postulate to have random dynamics, such as a hypothetical <u>card game</u> where we assume that the cards are truly shuffled into a random order.
2. **Incomplete observability**. Even deterministic systems can appear stochastic when we <u>cannot observe all of the variables that drive the behavior of the system</u>. For example, in the Monty Hall problem, a game show contestant is asked to choose between three doors and wins a prize held behind the chosen door. Two doors lead to a goat while a third leads to a car. The outcome given the contestant’s choice is deterministic, but from the contestant’s point of view, the outcome is uncertain.
3. **Incomplete modeling**. When we use a model that must discard some of the information we have observed, the discarded information results in uncertainty in the model’s predictions. For example, suppose we build a <u>robot</u> that can exactly observe the location of every object around it. If the robot discretizes space when predicting the future location of these objects,
  then the <u>discretization</u> makes the robot immediately become uncertain about the precise position of objects: each object could be anywhere within the discrete cell that it was observed to occupy.

In many cases, it is more practical to use a **simple but uncertain rule** rather than a **complex but certain one**, even if the true rule is deterministic and our modeling system has the fidelity to accommodate a complex rule. For example, the simple rule “Most birds fly” is cheap to develop and is broadly useful, while a rule of the form, “Birds fly, except for very young birds that have not yet learned to fly, sick or injured birds that have lost the ability to fly, flightless species of birds including the cassowary, ostrich and kiwi. . . ” is expensive to develop, maintain and communicate, and after all of this effort is still very brittle and prone to failure.

### Frequentist vs Bayesian 

Given that we need a means of <u>representing and reasoning about uncertainty</u>, it is not immediately obvious that probability theory can provide all of the tools we want for artificial intelligence applications. Probability theory was originally developed to analyze the frequencies of events. It is easy to see how probability theory can be used to study events like drawing a certain hand of cards in a game of poker. These kinds of events are often repeatable. When we say that an outcome has a probability p of occurring, it means that if we repeated the experiment (e.g., draw a hand of cards) infinitely many times, then proportion p
of the repetitions would result in that outcome. This kind of reasoning does not seem immediately applicable to propositions that are not repeatable. 

If a doctor analyzes a patient and says that the patient has a 40% chance of having the flu, this means something very different—we can not make infinitely many replicas of the patient, nor is there any reason to believe that different replicas of the patient
would present with the same symptoms yet have varying underlying conditions. In the case of the doctor diagnosing the patient, we use probability to represent a **degree of belief**, with 1 indicating absolute certainty that the patient has the flu and 0 indicating absolute certainty that the patient does not have the flu. 

The former kind of probability, <u>related directly to the rates at which events occur</u>, is known as **frequentist probability**, while the latter, <u>related to qualitative levels of certainty</u>, is known as **Bayesian probability**.

If we list several properties that we expect common sense reasoning about uncertainty to have, then <u>the only way to satisfy those properties is to treat Bayesian probabilities as behaving exactly the same as frequentist probabilities</u>. For example, if we want to compute the probability that a player will win a poker game given that she has a certain set of cards, we use exactly the same formulas as when we compute the probability that a patient has a disease given that she has certain symptoms. For more details about why a small set of common sense assumptions implies that the same axioms must control both kinds of probability, see Ramsey ( 1926 ).

Probability can be seen as the extension of logic to deal with uncertainty. Logic provides a set of formal rules for determining what propositions are implied to be true or false given the assumption that some other set of propositions is true or false. <u>Probability theory provides a set of formal rules for determining the likelihood of a proposition being true given the likelihood of other propositions.</u>

### Random Variable

A random variable is a variable that can take on different values randomly. We typically denote the random variable itself with a lower case letter in plain typeface x, and the values it can take on with lower case script letters. For example, x1 and x 2
are both possible values that the random variable x can take on. For vector-valued variables, we would write the random variable as **x** and one of its values as $x$.

### Probability Distributions

A probability distribution is a description of how likely a random variable or set of random variables is to take on each of its possible states.

#### Discrete Variables and Probability Mass Functions

A probability distribution over discrete variables may be described using a **probability mass function (PMF)**. We typically denote probability mass functions with a capital P . Often we associate each random variable with a different probabilitymass function and the reader must infer which probability mass function to use based on the identity of the random variable, rather than the name of the function; <u>P ( x ) is usually not the same as P ( y )</u> .

The probability mass function maps from a state of a random variable to the probability of that random variable taking on that state. The probability that x = $x$ is denoted as $P (x )$, with a probability of 1 indicating that x = $x$ is certain and a probability of 0 indicating that x = $x$ is impossible. Sometimes to disambiguate which PMF to use, we write the name of the random variable
explicitly: P (x = $x$). Sometimes we define a variable first, then use ∼ notation to specify which distribution it follows later: 

​	x ∼ P ( x ) .

Probability mass functions can act on many variables at the same time. Such a probability distribution over many variables is known as a **joint probability distribution**. P (x = $x$, y = $y$) denotes the probability that x = $x$ and y = $y$ simultaneously. We may also write $P ( x, y )$ for brevity.

#### Continuous Variables and Probability Density Functions

When working with continuous random variables, we describe probability distributions using a **probability density function (PDF)** rather than a probability mass function. To be a probability density function, a function p must satisfy the following properties:
• The domain of p must be the set of all possible states of x.
• ∀ x ∈ x, p ( x ) ≥ 0 . <u>Note that we do not require p ( x ) ≤ 1.</u>
• $\int p ( x ) dx = 1$.
A probability density function p(x) does not give the probability of a specific state directly, instead the probability of landing inside an infinitesimal region with volume δx is given by p ( x ) δx .

We can integrate the density function to find the actual probability mass of a set of points. Specifically, the probability that x lies in some set $\mathbb{S}$ is given by the integral of p (x ) over that set. In the univariate example, the probability that x lies in the interval [ a, b ] is given by $\int_{[ a,b ]} p ( x ) dx$ .

For an example of a probability density function corresponding to a specific probability density over a continuous random variable, consider a uniform distribution on an interval of the real numbers. We can do this with a function **u(x; a, b)**, where a and b are the endpoints of the interval, with b > a. <u>The “;” notation means “parametrized by”;</u> we consider x to be the argument of the function, while a and b are parameters that define the function. To ensure that there is no probability mass outside the interval, we say u(x; a, b) = 0 for all x ∈ [a, b] . Within [ a, b], $u ( x ; a, b ) = \frac {1} {b − a}$ . We can see that this is nonnegative everywhere. Additionally, it integrates to 1. We often denote that x follows the uniform distribution on [a, b] by writing **x ∼ U ( a, b )** .

### Marginal Probability

Sometimes we know the probability distribution over a set of variables and we want to know the probability distribution over just a subset of them. The probability distribution over the subset is known as the marginal probability distribution. For example, suppose we have discrete random variables x and y, and we know P (x , y ) . We can find P ( x ) with the sum rule :

$\forall x \in \texttt{x},  P(x) = \sum_y P(x,y)$ 

The name “marginal probability” comes from the process of computing marginal probabilities on paper. When the values of P (x , y ) are written in a grid with different values of x in rows and different values of y in columns, it is natural to sum across a row of the grid, then write P(x) in the margin of the paper just to the right of the row.

For continuous variables, we need to use integration instead of summation:

$p(x) = \int p(x,y)dy$

### Conditional Probability

In many cases, we are interested in the probability of some event, given that some other event has happened. This is called a conditional probability. We denote the conditional probability that y = $y$ given x = $x$ as P (y = $y$ | x = $x$). This conditional probability can be computed with the formula
$P ( \mathtt{y} = y | \mathtt{x} = x ) = \frac {P ( \mathtt{y} = y, \mathtt{x} = x )}  {P ( \mathtt{x} = x )}$
The conditional probability is only defined when P( x = x) > 0. We cannot compute the conditional probability conditioned on an event that never happens. 

It is important not to confuse conditional probability with computing what would happen if some action were undertaken. The conditional probability that a person is from Germany given that they speak German is quite high, but if a randomly selected person is taught to speak German, their country of origin does not change. Computing the consequences of an action is called making an intervention query. Intervention queries are the domain of causal modeling, which we do not explore in this book.

### The Chain Rule of Conditional Probabilities

Any joint probability distribution over many random variables may be decomposed into conditional distributions over only one variable:

$P(x^{(1)}...x^{(n)}) = p(x^{(1)})\prod_{i=2}^n P(x^{(i)}|P(x^{(1)}), ..., P(x^{(i-1)}))$

### Independence and Conditional Independence

Two random variables x and y are independent if their probability distribution can be expressed as a product of two factors, one involving only x and one involving only y:
$\forall x \in \texttt{x}, y \in \texttt{y} , p( x, y ) = p (x ) p (y ) $.
Two random variables x and y are conditionally independent given a random variable z if the conditional probability distribution over x and y factorizes in this way for every value of z:
$\forall x \in \texttt{x}, y \in \texttt{y}, z \in \texttt{z} , p( x,y,z ) = p ( x | z ) p ( y | z ) $.
We can denote independence and conditional independence with compact notation: x ⊥ y means that x and y are independent, while x ⊥ y | z means that x and y are conditionally independent given z.

### Expectation, Variance and Covariance

The **expectation** or **expected value** of <u>some function f (x ) with respect to a probability distribution P (x)</u> is the <u>average or mean value that f takes on when x is drawn from P</u> . For discrete variables this can be computed with a summation:

$\mathbb{E}_{X \sim P} f(x) = \sum_x f(x)P(x)$

While for continuous variables, it is computed with an integral:

$\mathbb{E}_{X \sim p} f(x) = \int_x f(x)p(x)dx$

When the identity of the distribution is clear from the context, we may simply write the name of the random variable that the expectation is over, as in $\mathbb{E}_x [f (x)]$. If it is clear which random variable the expectation is over, we may omit the subscript entirely, as in $\mathbb{E}[f (x)]$. By default, we can assume that E[·] averages over the values of all the random variables inside the brackets. Likewise, when there is no ambiguity, we may omit the square brackets.

Expectations are linear, for example,

$\mathbb{E}_x[\alpha f(x) + \beta g(x)] = \alpha \mathbb{E}_x[\mathbb{f(x)}] + \beta \mathbb{E}_x[g(x)]$

The **variance** gives a measure of how much the values of a function of a random variable x vary as we sample different values of x from its probability distribution:

$var(f(x)) = \mathbb{E} [(f(x) - \mathbb{E}[f(x)])^2]$

<u>When the variance is low, the values of f(x) cluster near their expected value.</u> The square root of the variance is known as the **standard deviation**.

The **covariance** gives some sense of <u>how much two values are linearly related to each other,</u> as well as the <u>scale of these variables</u>:

$cov(f(x), g(y)) = \mathbb{E} [(f(x) - \mathbb{E}[f(x)]) (g(y) - \mathbb{E}[g(y)]))]$

<u>High absolute values of the covariance mean that the values change very much and are both far from their respective means at the same time.</u> If the sign of the covariance is positive, then both variables tend to take on relatively high values
simultaneously. If the sign of the covariance is negative, then one variable tends to take on a relatively high value at the times that the other takes on a relatively low value and vice versa. 

Other measures such as **correlation** normalize the contribution of each variable in order to measure only how much the variables are related, rather than also being affected by the scale of the separate variables.

The notions of covariance and dependence are related, but are in fact distinct concepts. They are related because <u>two variables that are independent have zero covariance</u>, and <u>two variables that have non-zero covariance are dependent</u>. However, independence is a distinct property from covariance. For two variables to have zero covariance, there must be no linear dependence between them. <u>Independence is a stronger requirement than zero covariance</u>, because independence also excludes
nonlinear relationships. <u>It is possible for two variables to be dependent but have zero covariance</u>. 

The **covariance matrix** of a random vector x ∈ R n is an n × n matrix, such that $Cov(x)_{i,j} = Cov(x_i, x_j)$

### Bernoulli Distribution

The Bernoulli distribution is a distribution over a single binary random variable. It is controlled by a single parameter $\phi \in [0, 1]$, which gives the prbability of the random variable being equal to 1. It has the following properties:

$P(1) = \theta$

$P(0) = 1- \theta$

$P(x) = \theta^x + (1-\theta)^{1-x}$

$\mathbb{E}_x [x] = \theta$

$Var_x[x] = \theta(1-\theta)$

### Multinoulli Distribution

The multinoulli or categorical distribution is a distribution over a single discrete variable with k different states, where k is finite. The multinoulli distribution is parametrized by a vector $p \in [0, 1]^ {k−1}$ , where p i gives the probability of the i-th state. The final, k-th state’s probability is given by $1− 1 ^T p$. Note that we must constrain $1 ^T p ≤ 1$. Multinoulli distributions are often used to refer to distributions over categories of objects, so we do not usually assume that state 1 has numerical value 1, etc. For this reason, we do not usually need to compute the expectation or variance of multinoulli-distributed random variables.

The Bernoulli and multinoulli distributions are sufficient to describe any distribution over their domain. This is because they model discrete variables for which it is feasible to simply enumerate all of the states. When dealing with continuous variables, there are uncountably many states, so any distribution described by a small number of parameters must impose strict limits on the distribution.

### Gaussian Distribution

The most commonly used distribution over real numbers is the **normal distribution** , also known as the **Gaussian distribution ** $\mathcal{N}(x;\mu, \sigma^2) = \sqrt \frac {1} {2\pi \sigma^2} exp(-\frac {1} {2\sigma^2} (x-\mu)^2)$

![normal-distribution](normal-distribution.png)

The two parameters μ ∈ R and σ ∈ (0, ∞) control the normal distribution. The parameter μ gives the coordinate of the central peak. This is also the **mean** of the distribution: E[x] = μ. The **standard deviation** of the distribution is given by σ, and the **variance** by $\sigma^2$ .

When we evaluate the PDF, we need to square and invert σ. When we need to frequently evaluate the PDF with different parameter values, a more efficient way of parametrizing the distribution is to use a parameter β ∈ (0, ∞) to control the
**precision** or i**nverse variance** of the distribution:

$\mathcal{N}(x;\mu,\beta) = \sqrt {\frac {\beta} {2\pi} } exp(-\frac {1} {2} \beta (x-\mu)^2)$

Normal distributions are a sensible choice for many applications. In the absence of prior knowledge about what form a distribution over the real numbers should take, the normal distribution is a good default choice for two major reasons.

First, many distributions we wish to model are truly close to being normal distributions. <u>The central limit theorem shows that the sum of many independent random variables is approximately normally distributed.</u> This means that in practice, many complicated systems can be modeled successfully as normally distributed noise, even if the system can be decomposed into parts with more structured behavior.

Second, <u>out of all possible probability distributions with the same variance, the normal distribution encodes the maximum amount of uncertainty over the real numbers.</u> <u>We can thus think of the normal distribution as being the one that inserts the least amount of prior knowledge into a model.</u> Fully developing and justifying this idea requires more mathematical tools, and is postponed to Sec. 19.4.2.

#### multivariate normal distribution

The normal distribution generalizes to $R^n$ , in which case it is known as the **multivariate normal distribution**. It may be parametrized with a **positive definite symmetric matrix Σ** :

$\mathcal{N}(x;\mu,\Sigma) = \sqrt { \frac {1} {(2\pi)^n det(\Sigma)}} exp(-\frac {1} {2} (x-\mu)^T \Sigma^{-1} (x-\mu))$

The parameter μ still gives the **mean** of the distribution, though now it is **vector-valued**. The parameter Σ gives the **covariance** matrix of the distribution. As in the univariate case, when we wish to evaluate the PDF several times for many different values of the parameters, the covariance is not a computationally efficient way to parametrize the distribution, since we need to invert Σ to evaluate the PDF. We can instead use a **precision matrix β**:

$\mathcal{N}(x;\mu,\beta) = \sqrt { \frac {det(\beta)} {(2\pi)^n}} exp(-\frac {1} {2} (x-\mu)^T \beta (x-\mu))$

We often fix the covariance matrix to be a **diagonal matrix**. An even simpler version is the **isotropic Gaussian distribution**, whose covariance matrix is a scalar times the identity matrix.

### Exponential and Laplace Distributions

In the context of deep learning, we often want to have a probability distribution with a sharp point at x = 0. To accomplish this, we can use the exponential distribution:

$p(x;\lambda) = \lambda 1_{x>=0} exp(-\lambda x)$

The exponential distribution uses the indicator function $1_ {x≥0}$ to assign probability zero to all negative values of x .

A closely related probability distribution that allows us to place a sharp peak of probability mass at an arbitrary point μ is the Laplace distribution

$Laplace(x; \mu, \gamma) = \frac {1} {2\gamma}exp(-\frac {|(x-\mu)|} {\gamma})$

### The Dirac Distribution and Empirical Distribution

In some cases, we wish to specify that all of the mass in a probability distribution clusters around a single point. This can be accomplished by defining a PDF using the Dirac delta function, δ ( x ) : p ( x ) = δ ( x − μ ) .

The Dirac delta function is defined such that it is zero-valued everywhere except 0, yet integrates to 1. The Dirac delta function is not an ordinary function that associates each value x with a real-valued output, instead it is a different kind of mathematical object called a generalized function that is defined in terms of its properties when integrated. We can think of the Dirac delta function as being the limit point of a series of functions that put less and less mass on all points other than μ .

By defining p(x) to be δ shifted by −μ we obtain an infinitely narrow and infinitely high peak of probability mass where x = μ .
A common use of the Dirac delta distribution is as a component of an empirical distribution,

$\bar{p}(x) = \frac {1}{m} \sum_{i=1}^{m} \delta(x-x_i)$

which puts probability mass $\frac {1} {m}$ on each of the m points x (1) , . . . , x ( m ) forming a given data set or collection of samples. The Dirac delta distribution is only necessary to define the empirical distribution over continuous variables. For discrete variables, the situation is simpler: an empirical distribution can be conceptualized as a multinoulli distribution, with a probability associated to each possible input value that is simply equal to the empirical frequency of that value in the training set.

We can view the empirical distribution formed <u>from a dataset of training examples</u> as specifying the distribution that we sample from when we train a model on this dataset. Another important perspective on the empirical distribution is that it is the probability density that <u>maximizes the likelihood of the training data</u> (see Sec. 5.5 ).

### Mixtures of Distributions, Latent Variable

It is also common to define probability distributions by combining other simpler probability distributions. One common way of combining distributions is to construct a mixture distribution. <u>A mixture distribution is made up of several component distributions.</u> On each trial, the choice of which component distribution generates the sample is determined by sampling a component identity from a multinoulli distribution:

$P(x) = \sum_i p(c=i)p(x|c=i)$

where P ( c ) is the multinoulli distribution over component identities.

We have already seen one example of a mixture distribution: the empirical distribution over real-valued variables is a mixture distribution with one Dirac component for each training example.

The mixture model allows us to briefly glimpse a concept that will be of paramount importance later—the **latent variable** . A <u>latent variable is a random variable that we cannot observe directly.</u> The component identity variable c of the mixture model provides an example. Latent variables may be related to x through the joint distribution, in this case, P (x , c ) = P (x | c )P (c ). The distribution P (c) over the latent variable and the distribution P (x | c ) relating the latent variables to the visible variables determines the shape of the distribution P (x) even though it is possible to describe P (x) without reference to the latent variable. Latent variables are discussed further in Sec. 16.5 .

A very powerful and common type of mixture model is the Gaussian mixture model, in which the components p(x | c = i) are Gaussians. Each component has a separately parametrized mean μ ( i ) and covariance Σ ( i ) . Some mixtures can have more constraints. For example, the covariances could be shared across components via the constraint Σ ( i ) = Σ∀i. As with a single Gaussian distribution, the mixture of Gaussians might constrain the covariance matrix for each component to be diagonal or isotropic.

In addition to the means and covariances, the parameters of a Gaussian mixture specify the prior probability $\alpha_i = P (c = i)$ given to each component i. The word “prior” indicates that it expresses the model’s beliefs about c before it has observed x. By comparison, P( c | x) is a posterior probability, because it is computed after observation of x. A Gaussian mixture model is a universal approximator of densities, in the sense that any smooth density can be approximated with any specific, non-zero amount of error by a Gaussian mixture model with enough components.

![gaussian-mix](/gaussian-mix.png)

### sigmoid function

![logistic-sigmoid](logistic-sigmoid.png)

$\sigma(x) = \frac {1} {1+exp(-x)}$

The logistic sigmoid is <u>commonly used to produce the $\phi$ parameter of a Bernoulli distribution</u> because its range is (0,1), which lies within the valid range of values for the $\phi$ parameter. See Fig. 3.3 for a graph of the sigmoid function. The sigmoid function **saturates** when its argument is very positive or very negative, meaning that the function becomes very flat and insensitive to small changes in its input.

### softplus function

![soft-plus](/soft-plus.png)

$\zeta(x) = log(1+exp(x))$

The softplus function can be <u>useful for producing the β or σ parameter of a normal distribution</u> because its range is (0 , ∞ ). It also arises commonly when manipulating expressions involving sigmoids. The name of the softplus function comes from the
fact that it is a smoothed or “softened” version of 

$x^+ = max(0, x)$

### Bayes’ Rule

We often find ourselves in a situation where we know P ( y | x ) and need to know P ( x | y ). Fortunately, if we also know P (x), we can compute the desired quantity using Bayes’ rule :
$P(x|y) = \frac {P ( x ) P ( y | x )} {P(y)}$
$P(y) = \sum_x {P(x)P(y|x)}$

### Information Theory

#### Shannon entropy & self information

Information theory is a branch of applied mathematics that revolves around quantifying how much information is present in a signal. It was originally invented to study sending messages from discrete alphabets over a noisy channel, such as communication via radio transmission. In this context, information theory tells how to design optimal codes and calculate the expected length of messages sampled from specific probability distributions using various encoding schemes. In the context of
machine learning, we can also <u>apply information theory to continuous variables</u> where some of these message length interpretations do not apply. This field is fundamental to many areas of electrical engineering and computer science. In this
textbook, we mostly use a few key ideas from information theory to <u>characterize probability distribution</u>s or <u>quantify similarity between probability distributions</u>. 

The basic intuition behind information theory is that <u>learning that an unlikely event has occurred is more informative</u> than learning that a likely event has occurred. A message saying “the sun rose this morning” is so uninformative as to be unnecessary to send, but a message saying “there was a solar eclipse this morning” is very informative. We would like to quantify information in a way that formalizes this intuition. Specifically,
• Likely events should have low information content, and in the extreme case, events that are guaranteed to happen should have no information content whatsoever.
• Less likely events should have higher information content.
• Independent events should have additive information. For example, finding out that a tossed coin has come up as heads twice should convey twice as much information as finding out that a tossed coin has come up as heads once.

In order to satisfy all three of these properties, we define the **self-information** of an event x = $x$ to be
I ( x ) = − log P ( x ) .

In this book, we always use log to mean the natural logarithm, with base e. Our definition of I(x) is therefore written in units of **nats** . One nat is the amount of information gained by observing an event of probability $\frac {1} {e}$ . Other texts use <u>base-2 logarithms and units called **bits** or shannons</u> ; information measured in bits is just a rescaling of information measured in nats.

When x is continuous, we use the same definition of information by analogy, but some of the properties from the discrete case are lost. For example, an event with unit density still has zero information, despite not being an event that is guaranteed to occur.

Self-information deals only with a single outcome. We can quantify the amount of uncertainty in an entire probability distribution using the **Shannon entropy**:

$H(x) = \mathbb{E}_{x \sim P}[I(x)]$

also denoted H(P ). In other words, <u>the Shannon entropy of a distribution is the expected amount of information in an event drawn from that distribution.</u> It gives a lower bound on the number of bits (if the logarithm is base 2, otherwise the units
are different) needed on average to encode symbols drawn from a distribution P. Distributions that are nearly deterministic (where the outcome is nearly certain) have low entropy; <u>distributions that are closer to uniform have high entropy.</u> See
Fig. 3.5 for a demonstration. When x is continuous, the Shannon entropy is known as the <u>differential entropy</u>.

#### Kullback-Leibler (KL) divergence

If we have two separate probability distributions P ( x) and Q( x) over the same random variable x, we can measure how different these two distributions are using the Kullback-Leibler (KL) divergence:

$D_{KL}(P||Q) = H(Q) - H(P) = \mathbb{E}_{x \sim P}(logP(x) - logQ(x))$

In the case of discrete variables, it is the extra amount of information (measured in bits if we use the base 2 logarithm, but in machine learning we usually use nats and the natural logarithm) needed to send a message containing symbols drawn from probability distribution P , when we use a code that was designed to minimize the length of messages drawn from probability distribution Q .

#### Cross Entropy

A quantity that is closely related to the KL divergence is the cross-entropy $H (P, Q) = H (P ) + D_{KL} ( P || Q )$, which is similar to the KL divergence but lacking the term on the left:

$H(P,Q) = -\mathbb{E}_{x \sim P} logQ(x))$

Minimizing the cross-entropy with respect to Q is equivalent to minimizing the KL divergence, because Q does not participate in the omitted term.
