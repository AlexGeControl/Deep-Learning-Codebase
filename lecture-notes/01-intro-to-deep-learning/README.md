Week 01--Introduction to Deep Learning
1. Scale, both the depth of neural network and the size of dataset, drives the deeplearning progress.
2. In the regime of smaller training sets, the relative ordering of the algorithms is actually not very well defined.
So if you don't have a lot of training data is often up to your skill at feature engineering that determines the performance
3. Only in the regime of very large dataset does deep neural network dominates. 

Week 02--Shallow Neural Network
0. Linear hidden layer, or simple identity activation function, does not help as you go deeper in your network
1. Centralize / Normalize the data to have zero mean always help the optimization process of neural network
   So tanh should be preferred to sigmoid function.
2. The only exception should be in the output layer for binary classification. 
   In that situation, a mean of 0.5 is a good starting point for output probability.
3. One downside of both sigmoid and tanh activation functions is that their gradients all satuates when pre-activation 
   is significantly large than zero.
   That's the reason why ReLU is preferred in recent years.
4. Only use sigmoid for binary classification and only use linear activation for regression.
   At other situations, use ReLU or leaky ReLU
   The non-satuation property of ReLU will always boost the learning process of neural network.
   At last, the data and performance metrics finally tells.
5. Params of neural network should be carefully initialized.
   If params for all hidden units are the same, then only one type of hidden units for each layer will be trained.
   So the weight matrix for each layer should be randomly initialized.
   Weights should be initialized as small random values to avoid the saturation of activation function.
   When goes deep into the neural network, weight deviation should also be decreased.
   
