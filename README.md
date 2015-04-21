# bikeshare-ml
My machine learning implementations for the bikesharing competition on kaggle https://www.kaggle.com/c/bike-sharing-demand

The latest implementation of the algorithm and best is in neural 2

It's a neural network with sigmoid unction as its activation and the Y rescaled to between 0 and 1 so that it can be used by the neural network.

The errors in prediction are as follows. This is for the previous version of the algorithm.

JA =  5502.1
JB =    1.4910e+04
JAValidate =  4954.1
JBValidate =    1.3835e+04

The current version is the same but has 4 output units instead of one and averages them to get the final result. Furthermore the current algorithm does not have an activation function for the output layer. The errors in prediction are significantly better for this algorithm.

JA =  1534.7
JB =  3757.5
JAValidate =  1635.3
JBValidate =  3886.7
