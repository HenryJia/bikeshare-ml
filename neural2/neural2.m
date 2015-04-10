function [predictYaTrain, predictYbTrain, predictYaValidate, predictYbValidate, predictYaTest, predictYbTest] = neural2
%(plotX, X, y, alpha, lambda, iter)

% Load and normalise data

fprintf('Load Data\n');

Xa = load("trainP7_1.csv");
Xb = load("trainP7_2.csv");
Ya = load("trainYP2_1.csv" );
Yb = load("trainYP2_2.csv" );
validateXa = load("validateP7_1.csv");
validateXb = load("validateP7_2.csv");
validateYa = load("validateY1.csv");
validateYb = load("validateY2.csv");
testXa = load("original/testPF1_1.csv");
testXb = load("original/testPF1_2.csv");

% Set Important Variables:

alpha = 0.05;
lambda = 0;
iters = 20000;
plotIters = 40;

mTrainA = length(Ya);
mTrainB = length(Yb);

mValidateA = length(validateYa);
mValidateB = length(validateYb);

fprintf('Data Loaded. Normalised Features And Add Bias Units. Press Enter\n');
%pause;

Xa_norm =  featureRescale(Xa);
Xb_norm =  featureRescale(Xb);
validateXa_norm =  featureRescale(validateXa);
validateXb_norm =  featureRescale(validateXb);
testXa_norm =  featureRescale(testXa);
testXb_norm =  featureRescale(testXb);
Ya_norm = featureRescale(Ya) - 1;
Yb_norm = featureRescale(Yb) - 1;

%Add The Bias Units

fprintf('Features Normalised. Add Bias Units. Press Enter\n');
%pause;

Xa_norm = [ones(size(Xa_norm, 1) ,1), Xa_norm];
Xb_norm = [ones(size(Xb_norm, 1) ,1), Xb_norm];

validateXa_norm = [ones(size(validateXa_norm, 1) ,1), validateXa_norm];
validateXb_norm = [ones(size(validateXb_norm, 1) ,1), validateXb_norm];

testXa_norm = [ones(size(testXa_norm, 1) ,1), testXa_norm];
testXb_norm = [ones(size(testXb_norm, 1) ,1), testXb_norm];


fprintf('Features Normalised. Initialise Thetas. Press Enter\n');
%pause;

ThetaA1 = abs(randInitializeWeights(11, 22));
ThetaB1 = abs(randInitializeWeights(11, 22));

ThetaA2 = abs(randInitializeWeights(22, 1));
ThetaB2 = abs(randInitializeWeights(22, 1));

% Calculate Thetas & Results For First Hidden Layer

fprintf('Thetas initialise. Training. Press Enter\n');
%pause;

[ThetaA1, ThetaA2] = train(Xa_norm, Ya_norm, ThetaA1, ThetaA2, alpha, lambda, iters, plotIters);
[ThetaB1, ThetaB2] = train(Xa_norm, Ya_norm, ThetaB1, ThetaB2, alpha, lambda, iters, plotIters);

fprintf('Training Complete. Calculate Training Costs. Press Enter\n');
pause;

predictYaTrain = forwardPropagate(Xa_norm, ThetaA1, ThetaA2, Ya);
predictYbTrain = forwardPropagate(Xb_norm, ThetaB1, ThetaB2, Yb);

JA = ((predictYaTrain - Ya)' * (predictYaTrain - Ya) )/ (2 * mTrainA);
JB = ((predictYbTrain - Yb)' * (predictYbTrain - Yb) )/ (2 * mTrainB);

% Add on the penalty for regularization
JA += (sum(sum(ThetaA1(2:end, :) .^ 2)) + sum(sum(ThetaA1(2:end, :) .^ 2))) * (lambda / (2 * mTrainA))
JB += (sum(sum(ThetaB1(2:end, :) .^ 2)) + sum(sum(ThetaB1(2:end, :) .^ 2))) * (lambda / (2 * mTrainB))

figure(1)
scatter(1:mTrainA, Ya, "b")
hold on
plot(predictYaTrain, "r")
hold off

figure(2)
scatter(1:mTrainB, Yb, "b")
hold on
plot(predictYbTrain, "r")
hold off

figure(3)
scatter(Xa(:, 3), Ya, "b")
hold on
plot(Xa(:, 3), predictYaTrain, "r")
hold off

figure(4)
scatter(Xb(:, 3), Yb, "b")
hold on
plot(Xb(:, 3), predictYbTrain, "r")
hold off

predictYaValidate = forwardPropagate(validateXa_norm, ThetaA1, ThetaA2, Ya);
predictYbValidate = forwardPropagate(validateXb_norm, ThetaB1, ThetaB2, Yb);

JAValidate = ((predictYaValidate - validateYa)' * (predictYaValidate - validateYa) )/ (2 * mValidateA);
JBValidate = ((predictYbValidate - validateYb)' * (predictYbValidate - validateYb) )/ (2 * mValidateB);

% Add on the penalty for regularization
JAValidate += (sum(sum(ThetaA1(2:end, :) .^ 2)) + sum(sum(ThetaA1(2:end, :) .^ 2))) * (lambda / (2 * mValidateA))
JBValidate += (sum(sum(ThetaB1(2:end, :) .^ 2)) + sum(sum(ThetaB1(2:end, :) .^ 2))) * (lambda / (2 * mValidateB))

%fprintf('Calculated Final Results. Write To CSV. Press Enter\n');
%pause;

% Write To CSV

%csvwrite("result3_1.csv", testXa3);
%csvwrite("result3_2.csv", testXb3);

fprintf('Done. Press Enter\n');
%pause;