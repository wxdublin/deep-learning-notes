# Convolutional Networks

Convolutional networks ( LeCun , 1989 ), also known as **convolutional neural networks or CNNs** , are a specialized kind of neural network for processing **data that has a known, grid-like topology**. Examples include **time-series data**, which can be thought of as a 1D grid taking samples at regular time intervals, and **image data**, which can be thought of as a 2D grid of pixels. **Convolutional networks have been tremendously successful in practical applications.** The name “convolutional neural network” indicates that the network employs a mathematical operation called convolution. Convolution is a specialized kind of linear operation. Convolutional networks are simply <u>neural networks that use convolution in place of general matrix multiplication</u> in at least one of their layers.

## The Convolution Operation

Suppose we are tracking the location of a spaceship with a laser sensor. Our laser sensor provides a single output **x(t)**, the position of the spaceship at time t. Both x and t are real-valued, i.e., we can get a different reading from the laser sensor at any instant in time.

Now suppose that our laser sensor is somewhat noisy. <u>To obtain a less noisy estimate of the spaceship’s position, we would like to average together several measurements.</u> Of course, <u>more recent measurements are more relevant</u>, so we will want this to be a **weighted average that gives more weight to recent measurements.** We can do this with a **weighting function w(a)**, where a is the age of a measurement. If we apply such a weighted average operation at every moment, we obtain a new function s providing a smoothed estimate of the position of the spaceship:

$s(t) = \int x(a)w(t-a)da$

This operation is called convolution. The convolution operation is typically denoted with an asterisk:

$s(t) = (x * w)(t)$

In our example, **w needs to be a valid probability density function**, or the output is not a weighted average. Also, w needs to be 0 for all negative arguments, or it will look into the future, which is presumably beyond our capabilities. These limitations are particular to our example though. In general, convolution is defined for any functions for which the above integral is defined, and may be used for o<u>ther purposes besides taking weighted averages.</u>

In convolutional network terminology, the first argument (in this example, the function x) to the convolution is often referred to as the **input** and the second argument (in this example, the function w) as the **kernel** . The output is sometimes referred to as the **feature map.**

In our example, the idea of a laser sensor that can provide measurements at every instant in time is not realistic. Usually, when we work with data on a computer, time will be discretized, and our sensor will provide data at regular
intervals. In our example, it might be more realistic to assume that our laser provides a measurement once per second. The time index t can then take on only integer values. If we now assume that x and w are defined only on integer t, we can define the **discrete convolution**:

$s(t) = (x*w)(t) = \sum_{a=-\infty}^{+\infty} x(a) w(t-a)$

In machine learning applications, the **input** is usually a **multidimensional array of data** and the **kernel** is usually a **multidimensional array of parameters** that are adapted by the learning algorithm. We will refer to these multidimensional arrays as **tensors**. Because each element of the input and kernel must be explicitly stored
separately, we usually assume that these functions are zero everywhere but the finite set of points for which we store the values. This means that <u>in practice we can implement the infinite summation as a summation over a finite number of array elements.</u>

Finally, we often use convolutions over more than one axis at a time. For example, if we use a <u>two-dimensional image I as our input</u>, we probably also want to <u>use a two-dimensional kernel K</u> :

$S(i,j) = (I * K)(i,j) = \sum_m\sum_n I(m,n)K(i-m,j-n)$

Convolution is commutative $I*K = K*I$, but it is not usually an important property of a neural network implementation. Instead, many neural network libraries implement a related function called the **cross-correlation,** which is the same as convolution but without flipping the kernel:

$S(i,j) = (I*K)(i,j) = \sum_m\sum_n I(i+m, j+n)K(m, n)$

Many machine learning libraries implement cross-correlation but call it convolution. In this text we will follow this convention of calling both operations convolution, and specify whether we mean to flip the kernel or not in contexts where kernel flipping is relevant. In the context of machine learning, <u>the learning algorithm will learn the appropriate values of the kernel in the appropriate place</u>, so an algorithm based on convolution with kernel flipping will learn a kernel that is flipped relative to the kernel learned by an algorithm without the flipping. It is also rare for convolution to be used alone in machine learning; instead convolution is used simultaneously with other functions, and the combination of these functions does not commute regardless of whether the convolution operation flips its kernel or
not.

Discrete convolution can be viewed as multiplication by a matrix. However, the matrix has several entries constrained to be equal to other entries. For example, for univariate discrete convolution, each row of the matrix is constrained to be equal to the row above shifted by one element. This is known as a **Toeplitz matrix**. In two dimensions, a doubly block circulant matrix corresponds to convolution. In addition to these constraints that several elements be equal to each other, convolution usually corresponds to a very sparse matrix (a matrix whose entries are mostly equal to zero). This is because the kernel is usually much smaller than the input image. Any neural network algorithm that works with matrix multiplication and does not depend on specific properties of the matrix structure should work
with convolution, without requiring any further changes to the neural network. Typical convolutional neural networks do make use of further specializations in order to deal with large inputs efficiently, but these are not strictly necessary from a theoretical perspective.

![convolution](/home/wxu/proj2/deep-learning-notes/assets/convolution.png)

## Motivation

Convolution leverages three important ideas that can help improve a machine learning system: **sparse interactions** , **parameter sharing** and **equivariant representations**. Moreover, convolution provides **a means for working with inputs of variable size**. We now describe each of these ideas in turn.

### Sparse Interactions

**Traditional neural network layers** use matrix multiplication by a matrix of parameters with a separate parameter describing the interaction between each input unit and each output unit. This means **every output unit interacts with every input unit.** <u>Convolutional networks, however, typically have **sparse interactions** (also referred to as **sparse connectivity** or **sparse weights** ).</u> <u>This is accomplished by making the kernel smaller than the input.</u> For example, when processing an image, the input image might have thousands or millions of pixels, but we can detect <u>small, meaningful features</u> such as edges with kernels that occupy only tens or hundreds of pixels. This means that we need to store fewer parameters, which both reduces the memory requirements of the model and improves its statistical efficiency. It also means that computing the output requires fewer operations. These improvements in efficiency are usually quite large. If there are m inputs and n outputs, then matrix multiplication requires m × n parameters and the algorithms used in practice have O(m × n ) runtime (per example). If we limit the number of connections each output may have to k, then the sparsely connected approach requires only k × n parameters and O(k × n ) runtime. For many practical applications, it is possible to obtain good performance on the machine learning task while keeping k several orders of magnitude smaller than m . For graphical demonstrations of sparse connectivity, see Fig. 9.2 and Fig. 9.3 . <u>In a deep convolutional network, units in the deeper layers may indirectly interact with a larger portion of the input,</u> as shown in Fig. 9.4 . <u>This allows the network to efficiently describe complicated interactions between many variables by constructing such interactions from **simple building blocks** that each describe only sparse interactions.</u>

<u></u>![convolution2](/home/wxu/proj2/deep-learning-notes/assets/convolution2.png)

![convolution3](/home/wxu/proj2/deep-learning-notes/assets/convolution3.png)

### Parameter sharing

<u>Parameter sharing refers to using the same parameter for more than one function in a model.</u> In a traditional neural net, each element of the weight matrix is used exactly once when computing the output of a layer. It is multiplied by one element of the input and then never revisited. As a synonym for parameter sharing, one can say that a network has tied weights, because the value of the weight applied to one input is tied to the value of a weight applied elsewhere. In a convolutional neural net, each member of the kernel is used at every position of the input (except
perhaps some of the boundary pixels, depending on the design decisions regarding the boundary). The parameter sharing used by the convolution operation means that rather than learning a separate set of parameters for every location, we learn only one set. This does not affect the runtime of forward propagation—it is still O(k × n )—but it does further reduce the storage requirements of the model to k parameters. Recall that k is usually several orders of magnitude less than m. Since m and n are usually roughly the same size, k is practically insignificant compared to m × n . Convolution is thus dramatically more efficient than dense matrix multiplication in terms of the memory requirements and statistical efficiency. For a graphical depiction of how parameter sharing works, see Fig. 9.5 .

![convolution4](/home/wxu/proj2/deep-learning-notes/assets/convolution4.png)

As an example of both of these first two principles in action, Fig. 9.6 shows how sparse connectivity and parameter sharing can dramatically improve the efficiency of a linear function for detecting edges in an image.

![convolution5](/home/wxu/proj2/deep-learning-notes/assets/convolution5.png)

### equivariant representations

In the case of convolution, the particular form of parameter sharing causes the layer to have a property called **equivariance to translation.** T<u>o say a function is equivariant means that if the input changes, the output changes in the same way.</u> **Specifically, a function f(x) is equivariant to a function g if f (g(x)) = g(f(x)).** In the case of convolution, if we let g be any function that translates the input, i.e., shifts it, then the convolution function is equivariant to g. For example, let I be a function giving image brightness at integer coordinates. Let g be a function
mapping one image function to another image function, such that I' = g(I ) is the image function with I'(x, y) = I( x − 1, y). This shifts every pixel of I one unit to the right. If we apply this transformation to I , then apply convolution, the result will be the same as if we applied convolution to I'  , then applied the transformation g to the output. When processing time series data, this means that convolution produces a sort of timeline that shows when different features appear in the input. **If we move an event later in time in the input, the exact same representation of it will appear in the output, just later in time**. Similarly with images, convolution creates a 2-D map of where certain features appear in the input. I**f we move the object in the input, its representation will move the same amount in the output.** This is useful for when we know that some function of a small number of neighboring pixels is useful when applied to multiple input locations. For example, when processing images, it is useful to detect edges in the first layer of a convolutional network. The same edges appear more or less everywhere in the image, so it is practical to share parameters across the entire image. 

In some cases, we may not wish to share parameters across the entire image. For example, if we are processing images that are cropped to be centered on an individual’s face, we probably want to extract different features at different locations—the part of the network processing the top of the face needs to look for eyebrows, while the part of the network processing the bottom of the face needs to look for a chin.

<u>Convolution is not naturally equivariant to some other transformations, such as changes in the scale or rotation of an image. Other mechanisms are necessary for handling these kinds of transformations.</u>

### Input of variable size

Finally, some kinds of data cannot be processed by neural networks defined by matrix multiplication with a fixed-shape matrix. Convolution enables processing of some of these kinds of data. We discuss this further in Sec. 9.7 .

## Pooling

### Three Stages

A typical layer of a convolutional network consists of **three stages** (see Fig. 9.7 ). In the first stage, the layer performs several **convolutions** in parallel to produce a set of **linear activations**. In the second stage, each linear activation is run through a **nonlinear activation** function, such as the rectified linear activation function. This stage is sometimes called the **detector stage**. In the third stage, we use a **pooling function** to modify the output of the layer further.

![convolution6](/home/wxu/proj2/deep-learning-notes/assets/convolution6.png)

### Pooling Fuction

<u>A pooling function replaces the output of the net at a certain location with a **summary statistic** of the nearby outputs.</u> For example, the max pooling (Zhou and Chellappa , 1988 ) operation reports the <u>maximum output within a rectangular neighborhood</u>. Other popular pooling functions include <u>the average of a rectangular neighborhood,</u> the <u>L^2 norm of a rectangular neighborhood</u>, or a <u>weighted average based on the distance from the central pixel</u>.

####  **Invariance to translation** 

In all cases, pooling helps to make the representation become approximately invariant to small translations of the input. **Invariance to translation** means that if we translate the input by a small amount, the values of most of the pooled outputs do not change. See Fig. 9.8 for an example of how this works. <u>Invariance to local translation can be a very useful property if we care more about whether some feature is present than exactly where it is.</u> F**or example, when determining whether an image contains a face, we need not know the location of the eyes with pixel-perfect accuracy, we just need to know that there is an eye on the left side of the face and an eye on the right side of the face.** <u>In other contexts, it is more important to preserve the location of a feature. For example, if we want</u>
<u>to find a corner defined by two edges meeting at a specific orientation, we need to preserve the location of the edges well enough to test whether they meet.</u> **The use of pooling can be viewed as adding an infinitely strong prior that the function the layer learns must be invariant to small translations.** When this assumption is correct, it can greatly improve the statistical efficiency of the network. **Pooling over spatial regions produces invariance to translation, but if we pool over the outputs of separately parametrized convolutions, the features can learn which transformations to become invariant to (see Fig. 9.9 ).**

![convolution7](/home/wxu/proj2/deep-learning-notes/assets/convolution7.png)

![convolution7](/home/wxu/proj2/deep-learning-notes/assets/convolution8.png)

### Downsampling

<u>Because pooling summarizes the responses over a whole neighborhood, it is possible to use fewer pooling units than detector units,</u> by reporting summary statistics for pooling regions spaced k pixels apart rather than 1 pixel apart. See Fig. 9.10 for an example. This improves the computational efficiency of the network because the next layer has roughly k times fewer inputs to process. When the number of parameters in the next layer is a function of its input size (such as when the next layer is fully connected and based on matrix multiplication) this reduction in the input size can also result in improved statistical efficiency and reduced memory requirements for storing the parameters.

![convolution7](/home/wxu/proj2/deep-learning-notes/assets/convolution9.png)

### Handling inputs of varying size

**For many tasks, pooling is essential for handling inputs of varying size.** For example, if we want to classify images of variable size, the input to the classification layer must have a fixed size. <u>This is usually accomplished by varying the size of an offset between pooling regions so that the classification layer always receives the same number of summary statistics regardless of the input size.</u> <u>For example, the final pooling layer of the network may be defined to output four sets of summary statistics, one for each quadrant of an image, regardless of the image size.</u> Some theoretical work gives guidance as to which kinds of pooling one should use in various situations ( Boureau et al. , 2010 ). It is also possible to dynamically pool features together, for example, by running a clustering algorithm on the locations of interesting features ( Boureau et al. , 2011 ). This approach yields a different set of pooling regions for each image. Another approach is to learn a single pooling structure that is then applied to all images ( Jia et al. , 2012 ). Pooling can complicate some kinds of neural network architectures that use top-down information, such as Boltzmann machines and autoencoders. These issues will be discussed further when we present these types of networks in Part III. Pooling in convolutional Boltzmann machines is presented in Sec. 20.6 . The inverse-like operations on pooling units needed in some differentiable networks will be covered in Sec. 20.10.6 .

## complete CNN architecture

Some examples of complete convolutional network architectures for classification using convolution and pooling are shown in Fig. 9.11 .

![convolution7](/home/wxu/proj2/deep-learning-notes/assets/convolution10.png)

## Variants of the Basic Convolution Function

When discussing convolution in the context of neural networks, we usually do not refer exactly to the standard discrete convolution operation as it is usually understood in the mathematical literature. <u>The functions used in practice differ slightly.</u> Here we describe these differences in detail, and highlight some useful properties of the functions used in neural networks.

First, when we refer to convolution in the context of neural networks, we usually **actually mean an operation that consists of many applications of convolution in parallel.** <u>This is because convolution with a single kernel can only extract one kind of feature, albeit at many spatial locations. Usually we want each layer of our network to extract **many** kinds of **features**, at **many locations**.</u>

### Tensor

Additionally, the **input** is usually not just a grid of real values. Rather, it is a **grid of vector-valued observations**. For example, a color image has a red, green and blue intensity at each pixel. In a multilayer convolutional network, the input to the second layer is the output of the first layer, which usually has the output of many different convolutions at each position. When working with images, we usually think of the input and output of the convolution as being **3-D tensors**, with one index into the different channels and two indices into the spatial coordinates of each channel. Software implementations usually work in batch mode, so they will actually use **4-D tensors**, with the fourth axis indexing different **examples** in the batch, but we will omit the batch axis in our description here for simplicity.

Because convolutional networks usually use multi-channel convolution, the linear operations they are based on are not guaranteed to be commutative, even if kernel-flipping is used. These multi-channel operations are only commutative if each operation has the same number of output channels as input channels. Assume we have a 4-D kernel tensor K with element $K_{i,j,k,l}$ giving the **connection strength** between a unit in <u>channel i of the output</u> and a <u>unit in channel j of the input,</u> with an <u>offset of k rows and l columns</u> between the output unit and the input unit. Assume our input consists of **observed data V** with element $V_{i,j,k}$ giving the <u>value of the input unit within channel i at row j and column k.</u> Assume our **output** consists of Z with the same format as V . If **Z is produced by convolving K across V without flipping K** , then

$Z_{i,j,k} = \sum_{l,m,n} V_{l, j+m-1,k+n-1} K_{i,l,m,n}$

where the summation over l , m and n is over all values for which the tensor indexing operations inside the summation is valid. In linear algebra notation, we index into arrays using a 1 for the first entry. This necessitates the − 1 in the above formula. Programming languages such as C and Python index starting from 0 , rendering the above expression even simpler.

### strided convolution function

We may want to skip over some positions of the kernel in order to reduce the computational cost (at the expense of not extracting our features as finely). We can think of this as downsampling the output of the full convolution function. If we want to **sample only every s pixels** in each direction in the output, then we can define a **downsampled convolution function c** such that

$Z_{i,j,k} = c(V, K, s)_{i,j,k} = \sum_{l,m,n} V_{l, m+(j-1)s, n+(k-1)s} K_{i, l, m, n}$

<u>We refer to s as the **stride** of this downsampled convolution</u>. It is also possible to define a separate stride for each direction of motion. See Fig. 9.12 for an illustration.

![convolution7](/home/wxu/proj2/deep-learning-notes/assets/convolution11.png)

### Zero Padding

One essential feature of any convolutional network implementation is the ability to **implicitly zero-pad the input V in order to make it wider**. Without this feature, the width of the representation shrinks by one pixel less than the kernel width at each layer. Zero padding the input allows us to control the kernel width and the size of the output independently. Without zero padding, we are forced to choose between shrinking the spatial extent of the network rapidly and using small kernels—both scenarios that significantly limit the expressive power of the network. See Fig. 9.13 for an example.

![convolution7](/home/wxu/proj2/deep-learning-notes/assets/convolution12.png)

Three special cases of the zero-padding setting are worth mentioning. One is the extreme case in which no zero-padding is used whatsoever, and the convolution kernel is only allowed to visit positions where the entire kernel is contained entirely within the image. In MATLAB terminology, this is called **valid convolution**. In this case, all pixels in the output are a function of the same number of pixels in the input, so the behavior of an output pixel is somewhat more regular. However, the size of the output shrinks at each layer. If the input image has width m and the kernel has width k, the output will be of width m − k + 1. The rate of this shrinkage can be dramatic if the kernels used are large. Since the shrinkage is greater than 0, it limits the number of convolutional layers that can be included in the network. As layers are added, the spatial dimension of the network will eventually drop to 1 × 1, at which point additional layers cannot meaningfully be considered convolutional. Another special case of the zero-padding setting is when just enough zero-padding is added to keep the size of the output equal to the size of the input. MATLAB calls this **same convolution**. In this case, the network can contain as many convolutional layers as the available hardware can support, since the operation of convolution does not modify the architectural possibilities available to the next layer. However, the input pixels near the border influence fewer output pixels than the input pixels near the center. This can make the border pixels somewhat underrepresented in the model. This motivates the other extreme case, which MATLAB refers to as **full convolution**, in which enough zeroes are added for every pixel to be visited k times in each direction, resulting in an output image of <u>width m + k − 1</u> . In this case, the output pixels near the border are a function of fewer pixels than the output pixels near the center. This can make it difficult to learn a single kernel that performs well at all positions in the convolutional feature map. <u>Usually the optimal amount of zero padding (in</u>
<u>terms of test set classification accuracy) lies somewhere between “valid” and “same” convolution.</u>

### unshared convolution

In some cases, we do not actually want to use convolution, but rather **locally connected layers** ( LeCun , 1986 , 1989 ). In this case, the adjacency matrix in the graph of our MLP is the same, but <u>every connection has its own weight, specified by a 6-D tensor W</u> . The indices into W are respectively: i, the output channel, j, the output row, k, the output column, l, the input channel, m, the row offset within the input, and n, the column offset within the input. The linear part of a locally connected layer is then given by

$z_{i,j,k} = \sum_{l,m,n} [V_{l, j+m-1,k+n-1} w_{i,j,k,l,m,n}]$

This is sometimes also called **unshared convolution**, because it is a <u>similar operation to discrete convolution with a small kernel, but without sharing parameters across locations</u>. Fig. 9.14 compares **local connections, convolution, and full connections.**

![convolution7](/home/wxu/proj2/deep-learning-notes/assets/convolution13.png)

<u>Locally connected layers are useful when we know that each feature should be a function of a small part of space, but there is no reason to think that the same feature should occur across all of space. For example, if we want to tell if an image is a picture of a face, we only need to look for the mouth in the bottom half of the image.</u>

### Restricted Channel

It can also be useful to make versions of convolution or locally connected layers in which the connectivity is further restricted, for example to constrain that each output channel i be a function of only a subset of the input channels l. A common way to do this is to make the first m output channels connect to only the first n input channels, the second m output channels connect to only the second n input channels, and so on. See Fig. 9.15 for an example. Modeling interactions between few channels allows the network to have fewer parameters in order to reduce memory consumption and increase statistical efficiency, and also reduces the amount of computation needed to perform forward and back-propagation. It accomplishes these goals without reducing the number of hidden units.

![convolution7](/home/wxu/proj2/deep-learning-notes/assets/convolution14.png)

### Tiled convolution

Tiled convolution ( Gregor and LeCun , 2010a ; Le et al. , 2010 ) offers a **compromise between a convolutional layer and a locally connected layer**. Rather than learning a separate set of weights at every spatial location, we learn a **set of kernels that we rotate through as we move through space.** This means that immediately neighboring locations will have different filters, like in a locally connected layer, but the memory requirements for storing the parameters will increase only by a factor of the size of this set of kernels, rather than the size of the entire output feature map. See Fig. 9.16 for a comparison of locally connected layers, tiled convolution, and standard convolution.

![convolution7](/home/wxu/proj2/deep-learning-notes/assets/convolution15.png)

To define tiled convolution algebraically, let k be a 6-D tensor, where two of the dimensions correspond to different locations in the output map. Rather than having a separate index for each location in the output map, output locations cycle through a set of t different choices of kernel stack in each direction. If t is equal to the output width, this is the same as a locally connected layer.

$z_{i,j,k} = \sum_{l,m,n} V_{l, j+m-1,k+n-1} K_{i, j\%t+1, k\%t+1, l, m, n}$

where % is the modulo operation, with t%t = 0 , ( t + 1)%t = 1, etc. It is straightforward to generalize this equation to use a different tiling range for each dimension.

<u>Both locally connected layers and tiled convolutional layers have an interesting interaction with max-pooling: the detector units of these layers are driven by different filters. If these filters learn to detect different transformed versions of the same underlying features, then the max-pooled units become invariant to the learned transformation (see Fig. 9.9 ). Convolutional layers are hard-coded to be invariant specifically to translation.</u>

### Implementation

Other operations besides convolution are usually necessary to implement a convolutional network. To perform learning, one must be able to compute the gradient with respect to the kernel, given the gradient with respect to the outputs. In some simple cases, this operation can be performed using the convolution operation, but many cases of interest, including the case of stride greater than 1, do not have this property. 

Recall that convolution is a linear operation and can thus be described as a matrix multiplication (if we first reshape the input tensor into a flat vector). The matrix involved is a function of the convolution kernel. The matrix is sparse and each element of the kernel is copied to several elements of the matrix. This view helps us to derive some of the other operations needed to implement a convolutional network.

Multiplication by the transpose of the matrix defined by convolution is one such operation. This is the operation needed to back-propagate error derivatives through a convolutional layer, so it is needed to train convolutional networks that have more than one hidden layer. This same operation is also needed if we wish to reconstruct the visible units from the hidden units ( Simard et al. , 1992 ). **Reconstructing the visible units** is an operation commonly used in the models described in Part III of this book, such as autoencoders, RBMs, and sparse coding.

Transpose convolution is necessary to construct convolutional versions of those models. Like the kernel gradient operation, this input gradient operation can be implemented using a convolution in some cases, but in the general case requires a third operation to be implemented. <u>Care must be taken to coordinate this transpose operation with the forward propagation.</u> The size of the output that the transpose operation should return depends on the zero padding policy and stride of the forward propagation operation, as well as the size of the forward propagation’s output map. In some cases, multiple sizes of input to forward propagation can result in the same size of output map, so the transpose operation must be explicitly told what the size of the original input was.

These three operations—**convolution**, **backprop from output to weights,** and **backprop from output to inputs**—are sufficient to compute all of the gradients needed to train any depth of **feedforward convolutional network**, as well as to train convolutional networks with **reconstruction functions** based on the transpose of convolution. See Goodfellow ( 2010 ) for a full derivation of the equations in the **fully general multi-dimensional, multi-example case**. To give a sense of how these equations work, we present the two dimensional, single example version here.

Suppose **we want to train a convolutional network that incorporates strided convolution of kernel stack K applied to multi-channel image V with stride s as defined by c(K , V , s)** as:

  $Z_{i,j,k} = c(V, K, s)_{i,j,k} = \sum_{l,m,n} V_{l, m+(j-1)s, n+(k-1)s} K_{i, l, m, n}$

Suppose we want to minimize some loss function J(V , K ). <u>During forward propagation, we will need to use c itself to output Z</u> , <u>which is then propagated through the rest of the network and used to compute the cost function J</u>. 

<u>During back-propagation, we will receive a tensor G</u> such that 

$G_{i,j,k} = \frac {\partial} {\partial Z_{i,j,k}} J (V , K) $

To train the network, we need to compute the derivatives with respect to the weights in the kernel. To do so, we can use a function

![convolution7](/home/wxu/proj2/deep-learning-notes/assets/convolution16.png)

### Autoencoder networks

Autoencoder networks, described in Chapter 14 , are **feedforward networks trained to copy their input to their output**. A simple example is the **PCA algorithm**, that copies its input x to an approximate reconstruction r using the function $W^T W x$. It is common for more general autoencoders to **use multiplication by the transpose of the weight matrix** just as PCA does. To make such models convolutional, we can use the **function h to perform the transpose of the convolution operation**. Suppose we have hidden units H in the same format as Z and we define a reconstruction

$R=h(K, H, s)$

In order to <u>train the autoencoder</u>, we will receive the <u>gradient with respect to R as a tensor E</u> . To <u>train the decoder</u>, we need to obtain the <u>gradient with respect to K</u>. This is given by g(H , E , s). To <u>train the encoder</u>, we need to obtain
the <u>gradient with respect to H</u> . This is given by c(K , E , s). It is also possible to differentiate through g using c and h, but these operations are not needed for the back-propagation algorithm on any standard network architectures.

### bias

Generally, we do not use only a linear operation in order to transform from the inputs to the outputs in a convolutional layer. We generally also add some bias term to each output before applying the nonlinearity. This raises the question of how to share parameters among the biases. For locally connected layers it is natural to give each unit its own bias, and for tiled convolution, it is natural to share the biases with the same tiling pattern as the kernels. For convolutional layers, it is typical to have one bias per channel of the output and share it across all locations within each convolution map. However, if the input is of known, fixed size, it is also possible to learn a separate bias at each location of the output map. Separating the biases may slightly reduce the statistical efficiency of the model, but also allows the model to correct for differences in the image statistics at different locations. For example, when using implicit zero padding, detector units at the edge of the image receive less total input and may need larger biases.

## Structured Outputs

Convolutional networks can be used to output a high-dimensional, structured object, rather than just predicting a class label for a classification task or a real value for a regression task. Typically this object is just a tensor, emitted by a standard convolutional layer. For example, the model might emit a tensor S, where S i,j,k is the probability that pixel (j, k) of the input to the network belongs to class i. This allows the model to label every pixel in an image and draw precise masks that follow the outlines of individual objects.

## Data Types

## Efficient Convolution Algorithms