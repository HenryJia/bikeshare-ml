# bikeshare-ml
My machine learning implementations for the bikesharing competition on kaggle https://www.kaggle.com/c/bike-sharing-demand

The latest implementation of the algorithm and best is in neural 2

It's a neural network with sigmoid unction as its activation and the Y rescaled to between 0 and 1 so that it can be used by the neural network.

The errors in prediction are as follows. This is for the previous version of the algorithm. The current version is the same but has 4 output units instead of one and average s them to get the final result. The latest algorithm has much lower bias but has much higher variance.

JA =  5502.1
JB =    1.4910e+04
JAValidate =  4954.1
JBValidate =    1.3835e+04

These are the results of the latest algorithm

JA =  1914.6
JB =  4061.1
JAValidate =    1.2033e+04
JBValidate =    3.0502e+04
