# Sequence Modeling: Recurrent and Recursive Nets

Recurrent neural networks or RNNs ( Rumelhart et al. , 1986a ) are a family of neural networks for processing **sequential data**. Much as a convolutional network is a neural network that is specialized for processing a grid of values X such as an image, a recurrent neural network is a neural network that is specialized for processing a sequence of values x (1) , . . . , x ( τ ) . Just as convolutional networks can readily scale to images with large width and height, and some convolutional networks can process images of variable size, recurrent networks can scale to much longer sequences than would be practical for networks without sequence-based specialization. Most recurrent networks can also process sequences of variable length.

To go from multi-layer networks to recurrent networks, we need to take advantage of one of the early ideas found in machine learning and statistical models of the 1980s: **sharing parameters** across different parts of a model. Parameter sharing makes it possible to extend and apply the model to examples of different forms (different lengths, here) and generalize across them. If we had separate parameters for each value of the time index, we could not generalize to sequence lengths not seen during training, nor share statistical strength across different sequence lengths and across different positions in time. Such sharing is particularly important when a specific piece of information can occur at multiple positions within the sequence.

For example, consider the two sentences “I went to Nepal in 2009” and “In 2009, I went to Nepal.” If we ask a machine learning model to read each sentence and extract the year in which the narrator went to Nepal, we would like it to recognize the year 2009 as the relevant piece of information, whether it appears in the sixth word or the second word of the sentence. Suppose that we trained a feedforward network that processes sentences of fixed length. A traditional fully connected feedforward network would have separate parameters for each input feature, so it would need to learn all of the rules of the language separately at each position in the sentence. By comparison, **a recurrent neural network shares the same weights across several time steps.**

<u>A related idea is the use of convolution across a 1-D temporal sequence</u>. This convolutional approach is the basis for time-delay neural networks (Lang and Hinton , 1988 ; Waibel et al., 1989 ; Lang et al., 1990 ). <u>The convolution operation allows a network to share parameters across time, but is shallow.</u> The output of convolution is a sequence where each member of the output is a function of a **small number of neighboring members** of the input. The idea of parameter sharing manifests in the application of the same convolution kernel at each time step. 

Recurrent networks share parameters in a different way. **Each member of the output is a function of the previous members of the output.** Each member of the output is produced using the same update rule applied to the previous outputs. **This recurrent formulation results in the sharing of parameters through a very deep computational graph.**

For the simplicity of exposition, we refer to RNNs as operating on a sequence that contains vectors x ( t ) with the time step index t ranging from 1 to τ . In practice, recurrent networks usually operate on minibatches of such sequences, with a different sequence length τ for each member of the minibatch. We have omitted the minibatch indices to simplify notation. Moreover, the time step index need not literally refer to the passage of time in the real world, but only to the **position in the sequence**. RNNs may also be applied in two dimensions across spatial data such as images, and even when applied to data involving time, the network may have connections that go backwards in time, provided that the entire sequence is observed before it is provided to the network.

This chapter extends the idea of a **computational graph to include cycles**. These **cycles represent the influence of the present value of a variable on its own value at a future time step**. Such computational graphs allow us to define recurrent neural networks. We then describe many different ways to construct, train, and use recurrent neural networks.

For more information on recurrent neural networks than is available in this chapter, we refer the reader to the textbook of Graves ( 2012 ).

## Unfolding Computational Graphs

In this section we explain the idea of **unfolding** a recursive or recurrent computation into a computational graph that has a repetitive structure, typically corresponding to a chain of events. Unfolding this graph results in the **sharing of parameters** across a deep network structure.

For example, consider the classical form of a dynamical system:

$s^{(t)}= f(s^{(t-1)};\theta)$

where s ( t ) is called the state of the system. This equation is **recurrent** because the definition of s at time t refers back to the same definition at time t − 1 .

For a finite number of time steps τ , the graph can be unfolded by applying the definition τ − 1 times. For example, if we unfold it for τ = 3 time steps, we

$s^{(3)} = f(s^{(2)}; \theta) = f(f(s^{(1)};\theta);\theta)$

Unfolding the equation by repeatedly applying the definition in this way has yielded an expression that does not involve recurrence. Such an expression can now be represented by a **traditional directed acyclic computational graph**. The unfolded computational graph of Eq. 10.1 and Eq. 10.3 is illustrated in Fig. 10.1 .

![rnn1](/home/wxu/proj2/deep-learning-notes/assets/rnn1.png)

As another example, let us consider a dynamical system driven by an external signal x ( t ) ,

$s^{(t)} = f(s^{(t-1)}, x^{(t)}; \theta)$

Recurrent neural networks can be built in many different ways. Much as almost any function can be considered a feedforward neural network, **essentially any function involving recurrence can be considered a recurrent neural network.** Many recurrent neural networks use Eq. 10.5 or a similar equation to define the values of their hidden units. To indicate that the state is the hidden units of the network, we now rewrite Eq. 10.4 using the variable h to represent the state:

$h^{(t)} = f(h^{(t-1)}, x^{(t)}; \theta)$

illustrated in Fig. 10.2 , typical RNNs will add extra architectural features such as output layers that read information out of the state h to make predictions.

![rnn2](/home/wxu/proj2/deep-learning-notes/assets/rnn2.png)

When the recurrent network is trained to perform a task that requires predicting the future from the past, the network typically learns to use h ( t ) as a kind of lossy summary of the task-relevant aspects of the past sequence of inputs up to t. This summary is in general necessarily lossy, since it maps an arbitrary length sequence (x ( t ) , x ( t− 1) , x ( t− 2) , . . . , x (2) , x (1) ) to a fixed length vector h ( t ) . Depending on the training criterion, this summary might selectively keep some aspects of the past sequence with more precision than other aspects. For example, if the RNN is used in statistical language modeling, typically to predict the next word given previous words, it may not be necessary to store all of the information in the input sequence up to time t, but rather only enough information to predict the rest of the sentence. The most demanding situation is when we ask h ( t ) to be rich enough to allow
one to approximately recover the input sequence, as in autoencoder frameworks (Chapter 14 ).

Eq. 10.5 can be drawn in two different ways. One way to draw the RNN is with a diagram containing one node for every component that might exist in a physical implementation of the model, such as a biological neural network. In this view, the network defines a circuit that operates in real time, with physical parts whose current state can influence their future state, as in the left of Fig. 10.2 . Throughout this chapter, we use a black square in a circuit diagram to indicate that an interaction takes place with a delay of 1 time step, from the state at time t to the state at time t + 1. The other way to draw the RNN is as an unfolded computational graph, in which each component is represented by many different variables, with one variable per time step, representing the state of the component at that point in time. Each variable for each time step is drawn as a separate node of the computational graph, as in the right of Fig. 10.2 . What we call unfolding is the operation that maps a circuit as in the left side of the figure to a computational graph with repeated pieces as in the right side. The unfolded graph now has a size that depends on the sequence length.

We can represent the unfolded recurrence after t steps with a function g ( t ) :

$h^{(t)} = g^{(t)}(x^{(t)}, x^{(t-1)},...,x^{(1)}) = f(h^{(t-1)}, x^{(t)}; \theta)$

## Recurrent Neural Networks

Armed with the graph unrolling and parameter sharing ideas of Sec. 10.1 , we can design a wide variety of recurrent neural networks.

