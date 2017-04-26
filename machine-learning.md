---
typora-root-url: ./
---

## Learning Algorithms

### dataset, model, cost function, optimization

- dataset
  - training set: learn parameters, traning error
  - validation set: validate hyper parameter (weight decay, polynomial degree)
  - test set: generalization error
- model
  - capacity
  - hypothesis space, ploynomial degree
- cost fuction
  - regularization
- optimization: 
  - gradient, SGD
  - newton method

Machine learning is essentially a form of **applied statistics** with increased emphasis on the use of computers to **statistically estimate complicated functions** and a decreased emphasis on proving confidence intervals around these functions; we therefore present the two central approaches to statistics: **frequentist estimators and Bayesian inference**.

Most deep learning algorithms are based on an optimization algorithm called **stochastic gradient descent.** We describe how to combine various algorithm components such as an **optimization algorithm**, a **cost function**, a **model,** and a d**ataset** to build a machine learning algorithm. "“A computer program is said to learn from **experience E** with respect to some class of **tasks T**
and **performance measure P** , if its performance at tasks in T , as measured by P , improves with experience E .”

### The Task, T

**Machine learning allows us to tackle tasks that are too difficult to solve with fixed programs written and designed by human beings.** From a scientific and philosophical point of view, machine learning is interesting because developing our understanding of machine learning entails developing our understanding of the **principles that underlie intelligence**.

Machine learning tasks are usually described in terms of how the machine learning system should process an example . An **example** is a collection of **features** that have been quantitatively measured from some object or event that we want
the machine learning system to process. We typically represent an example as a vector $$x \in R^ n$$ where each entry $$x _i$$ of the vector is another feature. For example, the features of an image are usually the values of the pixels in the image.

Some of the most common machine learning tasks include the following:

- *Classification*: In this type of task, the computer program is asked to specify which of k categories some input belongs to. To solve this task, the learning algorithm is usually asked to produce a function $$f : \mathbb{R}^ n \rightarrow \{1, . . . , k\}$$. When y = f (x), the model assigns an input described by vector x to a category identified by numeric code y. There are other variants of the classification task, for example, where f outputs a probability distribution over classes. An example of a classification task is **object recognition**, where the input is an image (usually described as a set of pixel brightness values), and the output is a numeric code identifying the object in the image. Modern object recognition is best accomplished with deep learning.
- *Classification* with missing inputs: When some of the inputs may be missing, rather than providing a single classification function, the learning algorithm **must learn a set of functions**. Each function corresponds to classifying x with **a different subset of its inputs missing**. This kind of situation arises frequently in **medical diagnosis**, because many kinds of medical tests are expensive or invasive. One way to efficiently define such a large set of functions is to learn a **probability distribution** over all of the relevant variables, then solve the classification task by **marginalizing out the missing variables**. With n input variables, we can now **obtain all $$2 ^n$$ different classification functions** needed for each possible set of missing inputs, but we only need to learn a **single function describing the joint probability distribution**. See Goodfellow et al.
  ( 2013b ) for an example of a **deep probabilistic model** applied to such a task in this way. Many of the other tasks described in this section can also be generalized to work with missing inputs; classification with missing inputs is just one example of what machine learning can do.
- *Regression*: In this type of task, the computer program is asked to **predict a numerical value** given some input. To solve this task, the learning algorithm is asked to output a function $$f : R^ n \rightarrow R$$. This type of task is similar to classification, except that the format of output is different. An example of a regression task is the **prediction of the expected claim** amount that an 
  insured person will make (used to set insurance premiums), or the **prediction of future prices of securities.** These kinds of predictions are also used for **algorithmic trading**.
- *Transcription*: In this type of task, the machine learning system is asked to observe a relatively unstructured representation of some kind of data and transcribe it into discrete, textual form. For example, in **optical character recognition(OCR)**, the computer program is shown a photograph containing an image of text and is asked to return this text in the form of a sequence of characters (e.g., in ASCII or Unicode format). Google Street View uses deep learning to process address numbers in this way. Another example is **speech recognition**, where the computer program is provided an audio waveform and emits a sequence of characters or word ID codes describing the words that were spoken in the audio recording. Deep learning is a crucial component of modern speech recognition systems used at major companies including Microsoft, IBM and Google.
- *Machine translation*: In a machine translation task, the input already consists of a sequence of symbols in some language, and the computer program must convert this into a sequence of symbols in another language. This is commonly applied to **natural languages**, such as to translate from English to French. Deep learning has recently begun to have an important impact on this kind of task.
- *Structured output*: Structured output tasks involve any task where the output is a vector (or other data structure containing multiple values) with important relationships between the different elements. This is a broad category, and subsumes the transcription and translation tasks described above, but also many other tasks. One example is **parsing**—mapping a natural language sentence into a tree that describes its grammatical structure and tagging nodes of the trees as being verbs, nouns, or adverbs, and so on. See Collobert ( 2011 ) for an example of deep learning applied to a parsing task. Another example
  is **pixel-wise segmentation of images**, where the computer program assigns every pixel in an image to a specific category. For example, deep learning can be used to **annotate the locations of roads** in aerial photographs (Mnih and Hinton , 2010 ). The output need not have its form mirror the structure of the input as closely as in these annotation-style tasks. For example, in **image captioning**, the computer program observes an image and outputs a natural language sentence describing the image ( Kiros et al. , 2014a , b ; Mao et al. , 2015 ; Vinyals et al., 2015b ; Donahue et al., 2014 ; Karpathy and Li , 2015 ; Fang et al., 2015 ; Xu et al., 2015 ). These tasks are called structured output tasks because the program must output several values that are all tightly inter-related. For example, the words produced by an image captioning program must form a valid sentence.
- *Anomaly detection*: In this type of task, the computer program sifts through a set of events or objects, and flags some of them as being unusual or atypical. An example of an anomaly detection task is **credit card fraud detection**. By modeling your purchasing habits, a credit card company can detect misuse of your cards. If a thief steals your credit card or credit card information, the thief’s purchases will often come from a different probability distribution over purchase types than your own. The credit card company can prevent fraud by placing a hold on an account as soon as that card has been used
  for an uncharacteristic purchase. See Chandola et al. ( 2009 ) for a survey of anomaly detection methods.
- *Synthesis and sampling*: In this type of task, the machine learning algorithm is asked to generate new examples that are similar to those in the training data. Synthesis and sampling via machine learning can be useful for **media applications** where it can be expensive or boring for an artist to generate large volumes of content by hand. For example, **video games can automatically generate textures for large objects or landscapes**, rather than requiring an artist to manually label each pixel ( Luo et al. , 2013 ). In some cases, we want the sampling or synthesis procedure to generate some specific kind of
  output given the input. For example, in a **speech synthesis** task, we provide a written sentence and ask the program to emit an audio waveform containing a spoken version of that sentence. This is a kind of structured output task, but with the added qualification that **there is no single correct output for each input,** and we **explicitly desire a large amount of variation** in the output, in order for the output to seem more natural and realistic.
- *Imputation of missing values*: In this type of task, the machine learning algorithm is given a new example $$x \in R^ n$$ , but with some entries $$x_i$$ of x missing. The algorithm must provide a prediction of the values of the missing entries.
- *Denoising*: In this type of task, the machine learning algorithm is given in input a corrupted example $$\overline {x}  \in R^ n$$ obtained by an unknown corruption process from a clean example $$x \in R^ n$$ . The learner must predict the clean example x from its corrupted version $$\overline {x} $$, or more generally predict the conditional probability distribution $$p(x | \overline {x})$$.
- *Density estimation or probability mass function estimation* : In the density estimation problem, the machine learning algorithm is asked to **learn a function** $$p_{model} : R^ n \rightarrow R$$, where $$p_{model} (x)$$ can be interpreted as a p**robability density function** (if x is continuous) or a **probability mass function** (if x is discrete) on the space that the examples were drawn from. To do such a task well (we will specify exactly what that means when we discuss performance measures P ), the algorithm needs to **learn the structure of the data** it has seen. **It must know where examples cluster tightly and where they** **are unlikely to occur**. Most of the tasks described above require that the learning algorithm has at least implicitly captured the structure of the probability distribution. Density estimation allows us to explicitly capture that distribution. In principle, we can then perform computations on that distribution in order to solve the other tasks as well. For example, if we
  have performed density estimation to obtain a probability distribution p(x), we can use that distribution to solve the missing value imputation task. If a value x i is missing and all of the other values, denoted x −i , are given, then we know the distribution over it is given by p(x i | x −i ). In practice, density estimation does not always allow us to solve all of these related tasks, because in many cases the required operations on p(x) are computationally intractable. 

### The Performance Measure, P

In order to evaluate the abilities of a machine learning algorithm, we must design a quantitative measure of its performance.

For tasks such as **classification, classification with missing inputs, and transcription**, we often measure the accuracy of the model. **Accuracy** is just the proportion of examples for which the model produces the correct output. We can also obtain equivalent information by measuring the error rate, the proportion of examples for which the model produces an incorrect output. We often refer to the error rate as the expected 0-1 loss. The **0-1 loss** on a particular example is 0 if it is correctly classified and 1 if it is not. 

For tasks such as **density estimation**, it does not make sense to measure accuracy, error rate, or any other kind of 0-1 loss. Instead, we must use a different performance metric that gives the model a continuous-valued score for each example. The most common approach is to report the **average log-probability** the model assigns to some examples.

Usually we are interested in how well the machine learning algorithm performs on data that it has not seen before, since this determines how well it will work when deployed in the real world. We therefore evaluate these performance measures using a test set of data that is separate from the data used for training the machine learning system.

The choice of performance measure may seem straightforward and objective, but it is often difficult to choose a performance measure that corresponds well to the desired behavior of the system.

In some cases, this is because it is difficult to decide what should be measured. For example, when performing a transcription task, should we measure the accuracy of the system at transcribing entire sequences, or should we use a more fine-grained
performance measure that gives partial credit for getting some elements of the sequence correct? When performing a regression task, should we penalize the system more if it frequently makes medium-sized mistakes or if it rarely makes very large mistakes? These kinds of design choices depend on the application.

### The Experience, E

Machine learning algorithms can be broadly categorized as unsupervised or supervised by what kind of experience they are allowed to have during the learning process.

Most of the learning algorithms in this book can be understood as being allowed to experience an entire dataset . A dataset is a collection of many examples, as defined in Sec. 5.1.1 . Sometimes we will also call examples **data points**.

**Unsupervised learning algorithms** experience a dataset containing many features, then learn useful properties of the structure of this dataset. **In the context of deep learning, we usually want to learn the entire probability distribution that generated a dataset**, whether explicitly as in density estimation or implicitly for tasks like synthesis or denoising. Some other unsupervised learning algorithms perform other roles, like **clustering, which consists of dividing the dataset into clusters of similar examples.**

**Supervised learning** algorithms experience a dataset containing features, but each example is also associated with a **label** or **target** . 

Roughly speaking, unsupervised learning involves observing several examples of a random vector x, and attempting to implicitly or explicitly learn the probability distribution **p(x)**, or some interesting properties of that distribution, while supervised learning involves observing several examples of a random vector x and an associated value or vector y, and learning to predict y from x, usually by estimating **p(y | x )**. The term supervised learning originates from the view of the target y being provided by an instructor or teacher who shows the machine learning system what to do. In unsupervised learning, there is no instructor or
teacher, and the algorithm must learn to make sense of the data without this guide.

Unsupervised learning and supervised learning are not formally defined terms. The lines between them are often blurred. Many machine learning technologies can be used to perform both tasks. For example, the chain rule of probability states that for a vector $$x \in R ^n$$ , the joint distribution can be decomposed as

$p(x) = \prod_{i=1}^n p(x_i|x_{i-1},...x_1)$

This decomposition means that we can solve the ostensibly unsupervised problem of modeling p(x) by splitting it into n supervised learning problems. Alternatively, we can solve the supervised learning problem of learning p( y | x ) by using traditional unsupervised learning technologies to learn the joint distribution p(x, y) and inferring

$p(y|x) = \frac {p(x,y)} {\sum_{y'} p(x, y')}$

Though unsupervised learning and supervised learning are not completely formal or distinct concepts, they do help to roughly categorize some of the things we do with machine learning algorithms. Traditionally, people refer to **regression, classification**
**and structured output problems as supervised learning**. **Density estimation in support of other tasks is usually considered unsupervised learning.**

Some machine learning algorithms do not just experience a fixed dataset. For example, **reinforcement learning algorithms** interact with an environment, so there is a **feedback loop** between the learning system and its experiences. Such algorithms
are beyond the scope of this book. Please see Sutton and Barto ( 1998 ) or Bertsekas and Tsitsiklis ( 1996 ) for information about reinforcement learning, and **Mnih et al. ( 2013 ) for the deep learning approach to reinforcement learning.**

Most machine learning algorithms simply experience a dataset. A **dataset** can be described in many ways. In all cases, a dataset is a collection of **examples**, which are in turn collections of **features**.

One common way of describing a dataset is with a **design matrix** . A design matrix is a matrix containing a **different example in each row**. Each **column of the matrix corresponds to a different feature**. For heterogeneous data cases, rather than describing the dataset as a matrix with m rows, we will describe it as a set containing m elements:  $$\{x^{ (1)} , x^{ (2)} , . . . , x^{ ( m ) }\}$$. This notation does not imply that any two example vectors $$x^{(i)}$$ and $$x^ {( j )}$$ have the same size.

In the case of supervised learning, the example contains a label or target as well as a collection of features. For example, if we want to use a learning algorithm to perform object recognition from photographs, we need to specify which object
appears in each of the photos. We might do this with a numeric code, with 0 signifying a person, 1 signifying a car, 2 signifying a cat, etc. Often when working with a dataset containing a design matrix of feature observations X, we also provide a vector of labels y , with $$y_ i$$ providing the label for example i .

### Linear Regression Example

The goal is to build a system that can take a vector $$x ∈ R ^ n​$$ as input and predict the value of a scalar $$y \in R​$$ as its output. In the case of linear regression, the output is a linear function of the input. Let $$\hat{y}​$$ be the value that our model predicts y should take on. We define the output to be $\hat{y}=w^T x​$, where $w \in \mathbb{R}^n​$ is a vector of **parameters**.

Parameters are values that control the behavior of the system. In this case, w i is the coefficient that we multiply by feature x i before summing up the contributions from all the features. We can think of w as a set of **weights** that determine how each feature affects the prediction. If a feature $x_ i​$ receives a positive weight $w_i​$ ,then increasing the value of that feature increases the value of our prediction $\hat{y}​$ . If a feature receives a negative weight, then increasing the value of that feature decreases the value of our prediction. If a feature’s weight is large in magnitude, then it has a large effect on the prediction. If a feature’s weight is zero, it has no effect on the prediction.

We thus have a definition of our **task T : to predict y from x by outputting $\hat{y}= w ^T x​$**. 

Next we need a definition of our performance measure, P . 

Suppose that we have a **design matrix** of m example inputs that we will not use for training, only for evaluating how well the model performs. We also have a vector of regression targets providing the correct value of y for each of these examples. Because this dataset will only be used for evaluation, we call it the **test set.** We refer to the design matrix of inputs as $X^ {( test )}​$ and the vector of regression targets as $y^ {( test )}​$ . One way of measuring the performance of the model is to compute the mean squared error of the model on the test set. If $\hat{y} ^{( test )}​$ gives the predictions of the model on the test set, then the mean squared error is given by 

$MSE_{test} = \frac {1} {m} \| \hat{y} - y \|_2^2​$

so the error increases whenever the Euclidean distance between the predictions and the targets increases.

To make a machine learning algorithm, we need to design an algorithm that will <u>improve the weights w</u> in a way that reduces $MSE_{test}$ when the algorithm is allowed to gain experience by observing a training set $(X^ {( train )} , y^{ ( train )} )$. One intuitive way of doing this (which we will justify later, in Sec. 5.5.1 ) is just to minimize the mean squared error on the training set, $MSE_{train}$ .

![](/assets/normal-equation1.png)

![normal-equation2](/assets/normal-equation2.png)

The system of equations whose solution is given by Eq. 5.12 is known as the normal equations.

## Capacity, Overfitting and Underfitting

### Training Error & Generalization Error

The central challenge in machine learning is that we must perform well on new, previously unseen inputs—not just those on which our model was trained. The ability to perform well on previously unobserved inputs is called **generalization**.

Typically, when training a machine learning model, we have access to a training set, we can compute some error measure on the training set called the **training error**, and we reduce this training error. So far, what we have described is simply an optimization problem. What separates machine learning from optimization is that we want the **generalization error** , also called the **test error** , to be low as well. The generalization error is defined as the <u>expected value of the error on a new input</u>. Here the expectation is taken across different possible inputs, drawn from the distribution of inputs we expect the system to encounter in practice. We typically estimate the generalization error of a machine learning model by measuring its performance on a **test set** of examples that were collected separately from the training set.

How can we affect performance on the test set when we get to observe only the training set? The field of **statistical learning theory** provides some answers. If the training and the test set are collected arbitrarily, there is indeed little we can do. If we are allowed to make some assumptions about how the training and test set are collected, then we can make some progress. 

### Data Generating Distribution (i.i.d)

<u>The train and test data are generated by a probability distribution over datasets called the **data generating process**</u>. We typically make a set of assumptions known collectively as the **i.i.d. assumptions** These assumptions are that the examples
in each dataset are **independent** from each other, and that the train set and test set are **identically distributed**, drawn from the same probability distribution as each other. This assumption allows us to describe the data generating process with a probability distribution over a single example. The same distribution is then used to generate every train example and every test example. We call that shared underlying distribution the **data generating distribution, denoted** $p_{data}$ . <u>This probabilistic framework and the i.i.d. assumptions allow us to mathematically study the relationship between training error and test error.</u>

One immediate connection we can observe between the training and test error is that the <u>expected training error of a randomly selected model is equal to the expected test error</u> of that model. Suppose we have a probability distribution p(x, y) and we sample from it repeatedly to generate the train set and the test set. For some fixed value w, the expected training set error is exactly the same as the expected test set error, because both expectations are formed using the same dataset sampling process. The only difference between the two conditions is the name we assign to the dataset we sample.

Of course, when we use a machine learning algorithm, we do not fix the parameters ahead of time, then sample both datasets. We sample the training set, then use it to choose the parameters to reduce training set error, then sample the test set. Under this process, the expected test error is greater than or equal to the expected value of training error. The factors determining how well a machine learning algorithm will perform are its ability to:

1. Make the training error small.
2. Make the gap between training and test error small.

### Capacity, Underfitting, Overfitting

These two factors correspond to the two central challenges in machine learning: **underfitting and overfitting** . <u>Underfitting occurs when the model is not able to obtain a sufficiently low error value on the training set</u>. <u>Overfitting occurs when</u>
<u>the gap between the training error and test error is too large.</u> We can control whether a model is more likely to overfit or underfit by altering its **capacity**. Informally, <u>a model’s capacity is its ability to fit a wide variety of functions.</u> <u>Models with low capacity may struggle to fit the training set. Models with high capacity can overfit by memorizing properties of the training set that do not serve them well on the test set.</u>

### Hypothesis Space

One way to control the capacity of a learning algorithm is by choosing its **hypothesis space**, the set of functions that the learning algorithm is allowed to select as being the solution. For example, the linear regression algorithm has the set of all linear functions of its input as its hypothesis space. We can generalize linear regression to include **polynomials**, rather than just linear functions, in its hypothesis space. Doing so increases the model’s capacity.

A polynomial of degree one gives us the linear regression model with which we are already familiar, with prediction

$\hat{y} = b + wx​$

By introducing $x^2$ as another feature provided to the linear regression model, we can learn a model that is **quadratic** as a function of x :

$\hat{y} = b + w_1x + w_2 x^2​$

Though this model implements a quadratic function of its input, the output is still a linear function of the parameters, so we can still use the **normal equations** to train the model in **closed form**. We can continue to add more powers of x as additional features, for example to obtain a polynomial of degree 9:

$\hat{y} = b+\sum_{i=1}^{9}w_i x^i​$

<u>Machine learning algorithms will generally perform best when their capacity is appropriate in regard to the true complexity of the task they need to perform and the amount of training data they are provided with.</u> Models with insufficient capacity are unable to solve complex tasks. Models with high capacity can solve complex tasks, but when their capacity is higher than needed to solve the present task they may overfit.

Fig. 5.2 shows this principle in action. We compare a linear, quadratic and degree-9 predictor attempting to fit a problem where the true underlying function is quadratic. The linear function is unable to capture the curvature in the true underlying problem, so it underfits. The degree-9 predictor is capable of representing the correct function, but it is also capable of representing infinitely many other functions that pass exactly through the training points, because we have more parameters than training examples. We have little chance of choosing a solution that generalizes well when so many wildly different solutions exist. In this example, the quadratic model is perfectly matched to the true structure of the task so it generalizes well to new data.

![overfit-underfit](/assets/overfit-underfit.png)

### Occam’s razor

So far we have only described changing a model’s capacity by c<u>hanging the number of input features i</u>t has (and simultaneously adding new parameters associated with those features). There are in fact <u>many ways of changing a model’s capacity.</u> Capacity is not determined only by the choice of model. The model specifies which family of functions the learning algorithm can choose from when varying the parameters in order to reduce a training objective. This is called the **representational capacity** of the model. In many cases, finding the best function within this family is a very difficult optimization problem. In practice, the learning
algorithm <u>does not actually find the best function, but merely one that significantly reduces the training error.</u> These additional limitations, such as the imperfection of the optimization algorithm, mean that the learning algorithm’s **effective capacity**
may be less than the representational capacity of the model family.

Our modern ideas about <u>improving the generalization of machine learning models</u> are refinements of thought dating back to philosophers at least as early as Ptolemy. Many early scholars invoke a principle of parsimony that is now most widely known as **Occam’s razor** (c. 1287-1347). <u>This principle states that  among competing hypotheses that explain known observations equally well, one should choose the “simplest” one.</u> This idea was formalized and made more precise in the 20th century by the founders of <u>statistical learning theory</u> (Vapnik and Chervonenkis , 1971 ; Vapnik , 1982 ; Blumer et al., 1989 ; Vapnik , 1995 ).

### VC Dimension, Generalization Gap Bound

Statistical learning theory provides various means of <u>quantifying model capacity</u>. Among these, the most well-known is the Vapnik-Chervonenkis dimension, or **VC dimension**. The <u>VC dimension measures the capacity of a binary classifier</u>. The VC dimension is defined as being the <u>largest possible value of m for which there exists a training set of m different x points that the classifier can label arbitrarily.</u>

Quantifying the capacity of the model allows statistical learning theory to make quantitative predictions. The most important results in statistical learning theory show that <u>**the discrepancy between training error and generalization error is bounded**</u> from <u>above by a quantity that grows as the model capacity grows</u> but <u>shrinks as the number of training examples increases</u> (Vapnik and Chervonenkis, 1971 ; Vapnik , 1982 ; Blumer et al., 1989 ; Vapnik , 1995 ). <u>These bounds provide intellectual justification that machine learning algorithms can work, but they are rarely used in practice when working with deep learning algorithms.</u> This is in part because the bounds are often quite loose and in part because it can be <u>quite difficult to determine the capacity of deep learning algorithms.</u> The problem of determining the capacity of a deep learning model is especially difficult because the <u>effective capacity is limited by the capabilities of the optimization algorithm</u>, and we have l<u>ittle theoretical understanding of the very general non-convex optimization problems involved in deep learning.</u>

We must remember that while simpler functions are more likely to generalize (to have a small gap between training and test error) we must <u>still choose a sufficiently complex hypothesis to achieve low training error</u>. Typically, training error decreases until it <u>asymptotes to the minimum possible error value as model capacity increases</u> (assuming the error measure has a minimum value). Typically, generalization error has a U-shaped curve as a function of model capacity. This is illustrated in Fig. 5.3 .

![capacity-error](/assets/capacity-error.png)

### non-parametric models

To reach the most extreme case of arbitrarily high capacity, we introduce the concept of **non-parametric models**. So far, we have seen only parametric models, such as linear regression. <u>Parametric models learn a function described by a parameter vector whose size is finite and fixed before any data is observed.</u> Non-parametric models have no such limitation.

Sometimes, non-parametric models are just <u>theoretical abstractions</u> (such as an algorithm that searches over all possible probability distributions) that cannot be implemented in practice. However, we can also design practical <u>non-parametric</u>
<u>models</u> by making their <u>complexity a function of the training set size</u>. One example of such an algorithm is **nearest neighbor regression**. Unlike linear regression, which has a fixed-length vector of weights, <u>the nearest neighbor regression model simply</u>
<u>stores the X and y from the training set. When asked to classify a test point x, the model looks up the nearest entry in the training set and returns the associated regression target.</u> In other words, $\hat{y} = y_i​$ where $i = arg min \|X_{i,:} − x \| _2^2​$ . The algorithm can also be generalized to distance metrics other than the L 2 norm, such as learned distance metrics ( Goldberger et al. , 2005 ). If the algorithm is allowed to break ties by averaging the $y_i​$ values for all $X_{i,:}​$ that are tied for nearest, then this algorithm is able to achieve the minimum possible training error (which might be greater than zero, if two identical inputs are associated with different outputs) on any regression dataset.

Finally, we can also create a non-parametric learning algorithm by wrapping a parametric learning algorithm inside another algorithm that increases the number of parameters as needed. For example, we could imagine an <u>outer loop of learning</u>
<u>that changes the degree of the polynomial learned by linear regression on top of a polynomial expansion of the input.</u>

### Bayes error

The ideal model is an **oracle** that simply knows the true probability distribution that generates the data. Even such a model will still incur some error on many problems, because there may still be some **noise** in the distribution. In the case of supervised learning, the mapping from x to y may be **inherently stochastic**, or y may be a deterministic function that **involves other variables besides those included in x.** The error incurred by an oracle making predictions from the true distribution p (x , y ) is called the Bayes error.

Training and generalization error vary as the size of the training set varies. <u>**Expected generalization error** can never increase as the number of training examples increases</u>. For non-parametric models, more data yields better generalization until the best possible error is achieved. Any fixed parametric model with less than optimal capacity will asymptote to an error value that exceeds the Bayes error. See Fig. 5.4 for an illustration. <u>Note that it is possible for the model to have optimal capacity and yet still have a large gap between training and generalization error. In this situation, we may be able to reduce this gap by gathering more training examples.</u>

![training-size](/assets/training-size.png)

Figure 5.4: The effect of the training dataset size on the train and test error, as well as on the optimal model capacity. We constructed a synthetic regression problem based on adding moderate amount of noise to a degree 5 polynomial, generated a single test set, and then generated several different sizes of training set. For each size, we generated 40 different training sets in order to plot error bars showing 95% confidence intervals. (Top) The MSE on the train and test set for two different models: a quadratic model, and a model with degree chosen to minimize the test error. Both are fit in closed form. For the quadratic model, the training error increases as the size of the training set increases. This is because larger datasets are harder to fit. Simultaneously, the test error decreases, because fewer incorrect hypotheses are consistent with the training data. The quadratic
model does not have enough capacity to solve the task, so its test error asymptotes to a high value. The test error at optimal capacity asymptotes to the Bayes error. The training error can fall below the Bayes error, due to the ability of the training algorithm to memorize specific instances of the training set. As the training size increases to infinity, the training error of any fixed-capacity model (here, the quadratic model) must rise to at least the Bayes error. (Bottom) As the training set size increases, the optimal capacity (shown here as the degree of the optimal polynomial regressor) increases. The optimal capacity plateaus after reaching sufficient complexity to solve the task.

### The No Free Lunch Theorem

Learning theory claims that a machine learning algorithm can generalize well from a finite training set of examples. This seems to contradict some basic principles of logic. Inductive reasoning, or inferring general rules from a limited set of examples, is not logically valid. To logically infer a rule describing every member of a set, one must have information about every member of that set. In part, machine learning avoids this problem by offering only **probabilistic rules**, rather than the entirely certain rules used in purely logical reasoning. <u>Machine learning promises to find rules that are probably correct about most members of the set they concern.</u>

Unfortunately, even this does not resolve the entire problem. The **no free lunch theorem** for machine learning (Wolpert , 1996 ) states that, averaged over all possible data generating distributions, every classification algorithm has the same error rate when classifying previously unobserved points. In other words, in some sense, <u>no machine learning algorithm is universally any better than any other</u>. <u>The most sophisticated algorithm we can conceive of has the same average performance (over all possible tasks)</u> as merely predicting that every point belongs to the same class. Fortunately, these results hold only when we average over **all possible data generating distributions**. If we make assumptions about the kinds of probability distributions we encounter in real-world applications, then we can <u>design learning algorithms that perform well on **these** distributions.</u>

This means that the goal of machine learning research is <u>not to seek a **universal** learning algorithm</u> or the <u>absolute best learning algorithm.</u> Instead, our goal is to understand what kinds of distributions are relevant to the “real world” that an AI agent experiences, and what kinds of machine learning algorithms perform well on data drawn from the kinds of data generating distributions we care about.

### Regularization

The no free lunch theorem implies that we must design our machine learning algorithms to p**erform well on a specific task**. We do so by <u>building a set of preferences into the learning algorithm</u>. When these preferences are aligned with the learning problems we ask the algorithm to solve, it performs better.

So far, the only method of modifying a learning algorithm we have discussed is to <u>increase or decrease the model’s capacity by adding or removing functions from the hypothesis space</u> of solutions the learning algorithm is able to choose. We gave the specific example of increasing or decreasing the degree of a polynomial for a regression problem. The view we have described so far is oversimplified.

The behavior of our algorithm is strongly affected not just by how large we make the set of functions allowed in its hypothesis space, but by the **specific identity of those functions**. The learning algorithm we have studied so far, linear regression, has a hypothesis space consisting of the set of linear functions of its input. These linear functions can be very useful for problems where the relationship between inputs and outputs truly is close to linear. They are less useful for problems that behave in a very **nonlinear** fashion. For example, linear regression would not perform very well if we tried to use it to predict sin(x) from x. We can thus control the performance of our algorithms by choosing what kind of functions we allow them to draw solutions from, as well as by controlling the amount of these functions.

We can also give a learning algorithm a preference for one solution in its hypothesis space to another. This means that both functions are eligible, but one is preferred. The unpreferred solution be chosen only if it fits the training data significantly better than the preferred solution.

For example, we can modify the training criterion for linear regression to include **weight decay**. To perform linear regression with weight decay, we minimize a sum comprising both the mean squared error on the training and a criterion $J(w)$ that expresses a preference for the weights to have smaller squared L 2 norm. Specifically,

$J(w) = MSE_{train} + \lambda w^T w$

where λ is a value chosen ahead of time that controls the strength of our preference for smaller weights. When λ = 0, we impose no preference, and larger λ forces the weights to become smaller. <u>Minimizing $J (w )$ results in a choice of weights that make a tradeoff between fitting the training data and being small.</u> This gives us solutions that have a smaller slope, or put weight on fewer of the features. As an example of how we can control a model’s tendency to overfit or underfit via weight decay, we can train a high-degree polynomial regression model with different values of λ . See Fig. 5.5 for the results.

![regularization](/assets/regularization.png)

More generally, **we can regularize a model that learns a function $f(x; θ)$ by adding a penalty called a regularizer to the cost function.** In the case of weight decay, the regularizer is $\Omega(w) = w  ^Tw$. In Chapter 7 , we will see that many other regularizers are possible.

Expressing **preferences for one function over another** is a more general way of controlling a model’s capacity than including or excluding members from the hypothesis space. We can think of excluding a function from a hypothesis space as expressing an infinitely strong preference against that function.

In our weight decay example, we expressed our preference for linear functions defined with smaller weights explicitly, via an extra term in the criterion we minimize. There are many other ways of expressing preferences for different solutions, both implicitly and explicitly. Together, these different approaches are known as regularization. **Regularization is any modification we make to a learning algorithm that is intended to reduce its generalization error but not its training error.**  **Regularization** is one of the central concerns of the field of machine learning, rivaled in its importance only by **optimization**.

The no free lunch theorem has made it clear that there is no best machine learning algorithm, and, in particular, no best form of regularization. Instead we must choose a form of regularization that is well-suited to the particular task we want to solve. The philosophy of deep learning in general and this book in particular is that **a very wide range of tasks (such as all of the intellectual tasks that people can do) may all be solved effectively using very general-purpose forms of regularization.**

## Hyperparameters and Validation Sets

<u>Most machine learning algorithms have several settings that we can use to control the behavior of the learning algorithm. These settings are called **hyperparameters**.</u> The values of hyperparameters are not adapted by the learning algorithm itself
(though we can design a nested learning procedure where one learning algorithm learns the best hyperparameters for another learning algorithm).

In the polynomial regression example we saw in Fig. 5.2 , there is a single hyperparameter: **the degree of the polynomial**, which acts as a **capacity hyperparameter**. The **λ value** used to control the strength of **weight deca**y is another example of a
hyperparameter.

Sometimes a setting is chosen to be a hyperparameter that the learning algorithm does not learn because it is difficult to optimize. More frequently, we do not learn the hyperparameter because it is not appropriate to learn that hyperparameter on the training set. This applies to all hyperparameters that control model capacity. If learned on the training set, such hyperparameters would always choose the maximum possible model capacity, resulting in overfitting (refer to Fig. 5.3 ). For example, we can always fit the training set better with a higher degree polynomial and a weight decay setting of λ = 0 than we could with a lower degree polynomial and a positive weight decay setting.

To solve this problem, we need a **validation set** of examples that the training algorithm does not observe.

Earlier we discussed how a held-out **test set**, composed of examples coming from the same distribution as the training set, can be used to estimate the generalization error of a learner, after the learning process has completed. It is important that the test examples are not used in any way to make choices about the model, including its hyperparameters. For this reason, no example from the test set can be used in the validation set. Therefore, we always <u>construct the validation set from the training data</u>. <u>Specifically, we split the training data into two disjoint subsets. One of these subsets is used to learn the parameters. The other subset is our validation set, used to estimate the generalization error during or after training, allowing for the hyperparameters to be updated accordingly.</u> The subset of data used to learn the parameters is still typically called the **training set**, even though this may be confused with the larger pool of data used for the entire training process. <u>The subset of data used to guide the selection of hyperparameters is called the</u> **validation set**. Typically, one uses about 80% of the training data for training and
20% for validation. Since the validation set is used to “train” the hyperparameters, the validation set error will underestimate the generalization error, though typically by a smaller amount than the training error. After all hyperparameter optimization is complete, the generalization error may be estimated using the test set.

## Estimators, Bias and Variance

The field of **statistics** gives us many tools that can be used to achieve the machine learning goal of solving a task not only on the training set but also to generalize. F<u>oundational concepts such as parameter estimation, bias and variance are useful to formally characterize notions of generalization, underfitting and overfitting.</u>

### Point Estimation

Point estimation is the attempt to provide the single “best” prediction of some quantity of interest. In general the quantity of interest can be a single parameter or a vector of parameters in some parametric model, such as the weights in our linear regression example in Sec. 5.1.4 , but it can also be a whole function.

In order to distinguish estimates of parameters from their true value, our convention will be to denote a point estimate of a parameter θ by $\bar{θ}$.

Let $\{x^{(1)} , . . . , x^{( m )} \}$ be a set of m independent and identically distributed (i.i.d.) data points. A **point estimator** or **statistic** is any function of the data:

$\bar{\theta}_m = g(x^{(1)}, ...x^{(m)})$

The definition does not require that g return a value that is close to the true θ or even that the range of g is the same as the set of allowable values of θ. This definition of a point estimator is very general and allows the designer of an estimator great flexibility. While almost any function thus qualifies as an estimator, a good estimator is a function whose output is close to the true underlying θ that generated the training data.

For now, we take the **frequentist perspective on statistics**. That is, we assume that the true <u>parameter value θ is fixed but unknown</u>, while the <u>point estimate $\bar{θ}$ is a function of the data</u>. Since the data is drawn from a random process, any function of the data is random. Therefore $\bar{θ}$ is a random variable.

Point estimation can also refer to the estimation of the relationship between input and target variables. We refer to these types of point estimates as **function estimators**.

**Function Estimation** As we mentioned above, sometimes we are interested in performing function estimation (or function approximation). Here we are trying to predict a variable y given an input vector x . We assume that there is a function f(x) that describes the approximate relationship between y and x. For example, we may assume that $y = f(x) + \epsilon​$, where  $\epsilon​$ stands for the part of y that is not predictable from x. In function estimation, we are interested in approximating f with a model or estimate $\bar{f}​$ . Function estimation is really just the same as estimating a parameter θ; the function estimator $\bar{f}​$ is simply a point estimator in
function space. The linear regression example (discussed above in Sec. 5.1.4 ) and the polynomial regression example (discussed in Sec. 5.2 ) are both examples of scenarios that may be interpreted either as estimating a parameter w or estimating a function $\bar{f}$ mapping from x to y .

We now review the most commonly studied properties of point estimators and discuss what they tell us about these estimators.

### Bias

The bias of an estimator is defined as:

$bias(\bar{\theta}_m) = \mathbb{E} (\bar{\theta}_m) - \theta$

where the expectation is over the data (seen as samples from a random variable) and θ is the true underlying value of θ used to define the data generating distribution. An estimator $\bar{\theta}_m$ is said to be unbiased if bias($\bar{\theta}_m$ ) = 0, which implies that E( $\bar{\theta}_m$) = θ. An estimator $\bar{\theta}_m$is said to be asymptotically unbiased if $lim_{m \rightarrow \infty} bias(\bar{\theta}_m ) = 0$.

### Variance and Standard Error

Another property of the estimator that we might want to consider is how much we expect it to vary as a function of the data sample. Just as we computed the expectation of the estimator to determine its bias, we can compute its variance. The variance of an estimator is simply the variance $Var(\bar{\theta})$, where the r<u>andom variable is the training set</u>. Alternately, the square root of the
variance is called the standard error, denoted $SE( \bar{θ})$.

The variance or the standard error of an estimator provides a measure of how we would expect the estimate we compute from data to vary as we independently resample the dataset from the underlying data generating process. Just as we might like an estimator to exhibit low bias we would also like it to have relatively low variance.

### standard error of the mean 

When we compute any statistic using a finite number of samples, our **estimate of the true underlying parameter is uncertain**, in the sense that we could have obtained other samples from the same distribution and their statistics would have been different. The expected degree of variation in any estimator is a source of error that we want to quantify.

The standard error of the mean is given by

$SE(\bar{\mu}_m) = \sqrt {Var[\frac {1} {m}\sum_{i=1}^mx^{(i)}]} = \frac {\sigma} {\sqrt{m}}$

where $σ^2$ is the true variance of the samples $x^i$ . The standard error is often estimated by using an estimate of σ. Unfortunately, neither the square root of the sample variance nor the square root of the unbiased estimator of the variance provide an unbiased estimate of the standard deviation. Both approaches tend to underestimate the true standard deviation, but are still used in practice. The square root of the unbiased estimator of the variance is less of an underestimate. For large m , the approximation is quite reasonable.

The standard error of the mean is very useful in machine learning experiments. We often <u>estimate the generalization error by computing the sample mean of the error on the test set.</u> <u>The number of examples in the test set determines the accuracy of this estimate</u>. Taking advantage of the central limit theorem, which tells us that the mean will be approximately distributed with a normal distribution, we can use the standard error to compute the probability that the true expectation falls in any chosen interval. For example, the 95% confidence interval centered on the mean is $\bar\mu_m$ is 

$(\bar\mu_m-1.96SE(\bar\mu_m), \bar\mu_m+1.96SE(\bar\mu_m))$

under the normal distribution with mean μ and variance $SE(\bar\mu_m)^2$. In machine learning experiments, <u>it is common to say that algorithm A is better than algorithm B if the upper bound of the 95% confidence interval for the error of algorithm A is less than the lower bound of the 95% confidence interval for the error of algorithm B.</u>

### Trading off Bias and Variance to Minimize Mean Squared Error

<u>Bias and variance measure two different sources of error in an estimator.</u> Bias measures the expected deviation from the true value of the function or parameter. Variance on the other hand, provides a measure of the deviation from the expected estimator value that any particular sampling of the data is likely to cause. What happens when we are given a choice between two estimators, one with more bias and one with more variance? How do we choose between them? For example, imagine that we are interested in approximating the function shown in Fig. 5.2 and we are only offered the choice between a model with large bias and one that suffers from large variance. How do we choose between them?

<u>The most common way to negotiate this trade-off is to use **cross-validation**.</u> Empirically, cross-validation is highly successful on many real-world tasks. Alternatively, we can also compare the mean squared error (MSE) of the estimates:

$MSE = E[(\bar\theta_m-\theta)^2] = Bias(\bar\theta_m)^2 + Var(\bar\theta_m)$

The MSE measures the **overall expected deviation**—in a squared error sense— between the estimator and the true value of the parameter θ. As is clear from Eq. 5.54 , evaluating the MSE incorporates both the bias and the variance. Desirable estimators are those with small MSE and these are estimators that manage to keep both their bias and variance somewhat in check.

## Maximum Likelihood Estimation

Previously, we have seen some definitions of common estimators and analyzed their properties. But where did these estimators come from? Rather than guessing that some function might make a good estimator and then analyzing its bias and variance, we would like to have some principle from which we can derive specific functions that are good estimators for different models.

<u>The most common such principle is the **maximum likelihood principle**. Consider a set of m examples $X = \{x^{ (1)} , . . . , x ^{( m )}\}$ drawn independently from the true but unknown data generating distribution $p_{data} ( x )$ . Let $p_{model} (x; θ)$ be a parametric family of probability distributions over the same space indexed by θ. In other words, $p_{model} (x; θ)$maps any configuration x to a real number estimating the true probability $p_{data} ( x )$ .</u> The **maximum likelihood estimator** for θ is then defined as

$\theta_{ML} = arg max_\theta p_{model} (\mathbb{X; \theta}) = argmax_{\theta} \prod_{i=1}^m p_{model}(x^{(i)}; \theta)$

<u>This product over many probabilities can be inconvenient for a variety of reasons. For example, it is prone to numerical underflow. To obtain a more convenient but equivalent optimization problem, we observe that taking the logarithm of the likelihood does not change its arg max but does conveniently transform a product into a sum:</u>

$\theta_{ML} = argmax_{\theta}\sum_{i=1}^{m}log p_{model}(x^{(i)};\theta)$

Because the argmax does not change when we rescale the cost function, we can divide by m to obtain a version of the criterion that is expressed as an expectation with respect to the empirical distribution $\bar p_{data}$ defined by the training data:

$\theta_{ML} = argmax_{\theta}\mathbb{E_{x\sim \bar p_{data}}}(logp_{model}(x;\theta))$

One way to interpret maximum likelihood estimation is to view it as minimizing the dissimilarity between the empirical distribution $\bar p_{data}$ defined by the training set and the model distribution, with the degree of dissimilarity between the two
measured by the KL divergence. The KL divergence is given by

$D_{KL}(\bar p_{data}||p_{model}) = \mathbb{E}_{x \sim \bar p_{data}}[log\bar p_{data}(x)-log p_{model}(x)]$

The term on the left is a function only of the data generating process, not the model. This means when we train the model to minimize the KL divergence, we need only minimize

-$\mathbb{E_{x\sim \bar p_{data}}}(logp_{model}(x;\theta))$

Minimizing this KL divergence corresponds exactly to minimizing the cross-entropy between the distributions. Many authors use the term “cross-entropy” to identify specifically the negative log-likelihood of a Bernoulli or softmax distribution, but that is a misnomer. Any loss consisting of a negative log-likelihood is a cross entropy between the empirical distribution defined by the training set and the model. For example, mean squared error is the cross-entropy between the empirical distribution and a Gaussian model.

We can thus see maximum likelihood as an attempt to make the model distribution match the empirical distribution $\bar p_ {data}$ . Ideally, we would like to match the true data generating distribution p data , but we have no direct access to this distribution. 

While the optimal θ is the same regardless of whether we are maximizing the likelihood or minimizing the KL divergence, the values of the objective functions are different. In software, we often phrase both as minimizing a cost function. Maximum likelihood thus becomes minimization of the negative log-likelihood (NLL), or equivalently, minimization of the cross entropy. The perspective of maximum likelihood as minimum KL divergence becomes helpful in this case because the KL divergence has a known minimum value of zero. The negative log-likelihood can actually become negative when x is real-valued.

### Conditional Log-Likelihood and Mean Squared Error

The maximum likelihood estimator can readily be generalized to the case where our goal is to estimate a conditional probability P ( y | x ; θ) in order to predict y given x. <u>This is actually the most common situation because it forms the basis for most supervised learning.</u> If X represents all our inputs and Y all our observed targets, then the conditional maximum likelihood estimator is

$\theta_{ML} = argmax_{\theta} P(Y|X;\theta)$

If the examples are assumed to be i.i.d., then this can be decomposed into

$\theta_{ML} = argmax_{\theta} \sum_{i=1}^{m}logP(y^{(i)}|x^{(i)};\theta)$

### Example: Linear Regression as Maximum Likelihood

Linear regression, introduced earlier in Sec. 5.1.4 , may be justified as a maximum likelihood procedure. Previously, we motivated linear regression as an algorithm that learns to take an input x and produce an output value $\bar y$ . **The mapping from x to $\bar y$ is chosen to minimize mean squared error, a criterion that we introduced more or less arbitrarily.** We now revisit linear regression from the point of view of maximum likelihood estimation. Instead of producing a single prediction $\bar y$ , **we now think of the model as producing a conditional distribution p(y | x).** We can imagine that <u>with an infinitely large training set, we might see several training examples with the same input value x but different values of y</u> . The goal of the learning algorithm is now to fit the distribution p(y | x) to all of those different y values that are all compatible with x. To derive the same linear regression algorithm we obtained before, we define 

$p( y | x ) = \mathcal{N} (y; \bar y(x; w), \sigma^2 )$. 

<u>The function $\bar y(x; w )$ gives the prediction of the mean of the Gaussian.</u> In this example, we assume that the variance is fixed to
some constant $\sigma^2$ chosen by the user. We will see that this choice of the functional form of p(y | x) causes the maximum likelihood estimation procedure to yield the same learning algorithm as we developed before. Since the examples are assumed
to be i.i.d., the conditional log-likelihood (Eq. 5.63 ) is given by

![ml-linear-regression](/assets/ml-linear-regression.png)

where $\hat{y}( i )$ is the output of the linear regression on the i-th input $x^{( i )}$ and m is the number of the training examples. Comparing the log-likelihood with the mean squared error, we immediately see that maximizing the log-likelihood with respect to w yields
the same estimate of the parameters w as does minimizing the mean squared error.

The two criteria have different values but the same location of the optimum. This justifies the use of the MSE as a maximum likelihood estimation procedure. As we will see, the maximum likelihood estimator has several desirable properties.

## Bayesian Statistics

So far we have discussed **frequentist statistics** and approaches based on estimating a single value of θ, then making all predictions thereafter based on that one estimate. Another approach is to <u>consider all possible values of θ</u> when making a prediction. The latter is the domain of **Bayesian statistics** .

As discussed in Sec. 5.4.1 , <u>the frequentist perspective is that the true parameter value θ is fixed but unknown, while the point estimate $\hat θ​$ is a random variable on account of it being a function of the dataset (which is seen as random).</u>

The Bayesian perspective on statistics is quite different. The Bayesian <u>uses probability to reflect degrees of certainty of states of knowledge</u>. <u>The dataset is directly observed and so is not random.</u> <u>On the other hand, the true parameter θ is unknown or uncertain and thus is represented as a random variable.</u>

Before observing the data, we represent our knowledge of θ using the prior probability distribution, p(θ ) (sometimes referred to as simply “the prior”). Generally, the machine learning practitioner selects a prior distribution that is quite broad (i.e. with high entropy) to reflect a high degree of uncertainty in the value of θ before observing any data. For example, one might assume a priori that θ lies in some finite range or volume, with a uniform distribution. Many priors instead reflect a preference for “simpler” solutions (such as smaller magnitude coefficients, or a function that is closer to being constant).

Now consider that we have a set of data samples $\{x^{(1)}, . . . , x ^{(m)}\}$. We can recover the effect of data on our belief about θ by combining the data likelihood $p ( x,^{(1)} . . . , x  ^{(m)}| θ) $with the prior via Bayes’ rule:

$p(\theta|x^{(1)},...,x^{(m)}) = \frac {p(\theta)p(x^{(1)},...,x^{(m)}|\theta))} {p(x^{(1)},...,x^{(m)})}$

In the scenarios where Bayesian estimation is typically used, <u>the prior begins as a relatively uniform or Gaussian distribution with high entropy, and the observation of the data usually causes the posterior to lose entropy and concentrate around a few highly likely values of the parameters.</u>

Relative to maximum likelihood estimation, Bayesian estimation offers two important differences. First, unlike the maximum likelihood approach that makes predictions using a point estimate of θ, the Bayesian approach is to make predictions using a full distribution over θ. For example, after observing m examples, the predicted distribution over the next data sample, $x^{(m+1)}$, is given by

$p(x^{(m+1)}|x^{(1)},...,x^{(m)}) = \int p(\theta|x^{(1)},...,x^{(m)}) p(x^{(m+1)}|\theta)d\theta$

Here each value of θ with positive probability density contributes to the prediction of the next example, with the contribution weighted by the posterior density itself. After having observed $\{x^{(1)} , . . . , x^{(m)} \}$, if we are still quite uncertain about the
value of θ, then this uncertainty is incorporated directly into any predictions we might make.

In Sec. 5.4 , we discussed <u>how the frequentist approach addresses the uncertainty in a given point estimate of θ by evaluating its variance. The variance of the estimator is an assessment of how the estimate might change with alternative samplings of the observed data.</u> <u>The Bayesian answer to the question of how to deal with the uncertainty in the estimator is to simply integrate over it, which tends to protect well against overfitting.</u> This integral is of course just an application of the laws of probability, making the Bayesian approach simple to justify, while the frequentist machinery for constructing an estimator is based on the rather ad hoc decision to summarize all knowledge contained in the dataset with a single point estimate.

The second important difference between the Bayesian approach to estimation and the maximum likelihood approach is due to the contribution of the Bayesian prior distribution. The prior has an influence by shifting probability mass density towards regions of the parameter space that are preferred a priori . In practice, the prior often expresses a preference for models that are simpler or more smooth. Critics of the Bayesian approach identify the prior as a source of subjective human judgment impacting the predictions.

**Bayesian methods typically generalize much better when limited training data is available, but typically suffer from high computational cost when the number of training examples is large.**

### Example: Bayesian Linear Regression 

Here we consider the Bayesian estimation approach to learning the linear regression parameters. In linear regression, we learn a linear mapping from an input vector $x \in R^n$ to predict the value of a scalar y ∈ R . The prediction is parametrized by the vector $w \in R^n$ : 

$\hat y = w^T x$

Given a set of m training samples $(X^{( train )} , y^{( train )} )$, we can express the prediction of y over the entire training set as:

$\hat y^{(train)} = X^{(train)}w$

