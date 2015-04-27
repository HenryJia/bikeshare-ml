# bikeshare-ml
My machine learning implementations for the bikesharing competition on kaggle https://www.kaggle.com/c/bike-sharing-demand

The latest implementation of the algorithm and best is in neural 2

It's a neural network with natural logarithm as its activation and the Y rescaled to between 0 and 1 so that it can be used by the neural network.

The errors in prediction are as follows

JA =  5903.7
JB =    1.5014e+04
JAValidate =  5258.1
JBValidate =    1.4295e+04

An version 1 of the sigmoidNeural branch -> sigmoid function activated neural network, 3 layers

JA = 5502.1 JB = 1.4910e+04 JAValidate = 4954.1 JBValidate = 1.3835e+04

The version 2 of the sigmoidNeural branch -> same as the version 1 but the output unit is not activated, 3 layers

JA = 1534.7 JAValidate = 1635.3 JB = 3757.5 JBValidate = 3886.7

The version 1 of the sigmoidNeural3+ branch -> sigmoid activated neural network just like the version 2 of the sigmoidNeural branch but with more layers than 3

JA = 562.68 JAValidate = 743.38 JB = 1071.0 JBValidate = 1370.4

The version 2 of the sigmoidNeural3+ branch -> sigmoid activated neural network just like the version 2 of the sigmoidNeural3+ branch but with 16 output units instead of 1. The end result of the output units are averaged.

JA = 330.74 JAValidate = 666.86 JB = 673.13 JBValidate = 1222.8

The version 3 of the sigmoidNeural3+ branch -> Same as version 2 but bigger and also with one more input layer.

JA = 223.8534764672244 JAValidate = 670.2424706468234 JB = 504.2598015271401 JBValidate = 1160.346418306346

However, due to the fact that version 3 was overfitting the training set, I rolled it back to version 2 on the sigmoidNeural3+ branch.

Due to the fact that version 2 on the sigmoidNeural3+ branch is the best workign alfgorithm, it has been merged into the master branch.

Instructions:

To run the algorithm, execute the function in neural2/neural2_1.m and neural2/neural2_2.m

The scripts will then simulate a 4 layer neural network and calcualte the errors on the training and validation set and produce the predictions for kaggle's test set.