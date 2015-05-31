function [ThetaB1, ThetaB2, ThetaB3] = neural2_2

% Load and normalise data

fprintf('Load Data\n');

Xa = load("trainP7_1.csv")(:, [1, 3:end]);
Xb = load("trainP7_2.csv")(:, [1, 3:end]);
Ya = load("trainYP2_1.csv" );
Yb = load("trainYP2_2.csv" );
validateXa = load("validateP7_1.csv")(:, [1, 3:end]);
validateXb = load("validateP7_2.csv")(:, [1, 3:end]);
validateYa = load("validateY1.csv");
validateYb = load("validateY2.csv");
testXa = load("original/testPF1_1.csv")(:, [1, 3:end]);
testXb = load("original/testPF1_2.csv")(:, [1, 3:end]);

% Set Important Variables:

alpha = 0.00025;
lambda = 0;
iters = 100000;
scatterIters = 1000;

mTrainA = length(Ya);
mTrainB = length(Yb);

mValidateA = length(validateYa);
mValidateB = length(validateYb);

fprintf('Data Loaded. Normalised Features And Add Bias Units. Press Enter\n');

Xa_norm =  featureNormalize(Xa);
Xb_norm =  featureNormalize(Xb);
validateXa_norm =  featureNormalize(validateXa);
validateXb_norm =  featureNormalize(validateXb);
testXa_norm =  featureNormalize(testXa);
testXb_norm =  featureNormalize(testXb);

%Add The Bias Units

fprintf('Features Normalised. Add Bias Units. Press Enter\n');

Xa_norm = [ones(size(Xa_norm, 1) ,1), Xa_norm];
validateXa_norm = [ones(size(validateXa_norm, 1) ,1), validateXa_norm];
testXa_norm = [ones(size(testXa_norm, 1) ,1), testXa_norm];

fprintf('Features Normalised. Initialise Thetas. Press Enter\n');

ThetaA1 = abs(randInitializeWeights(10, 40));
ThetaA2 = abs(randInitializeWeights(40, 160));
ThetaA3 = abs(randInitializeWeights(160, 1));

% Calculate Thetas & Results For First Hidden Layer

fprintf('Thetas initialise. Training. Press Enter\n');

tic
[ThetaB1, ThetaB2, ThetaB3] = train(Xb_norm, Yb, ThetaB1, ThetaB2, ThetaB3, alpha, lambda, iters, scatterIters);
toc

fprintf('Training Complete. Calculate Costs. Press Enter\n');
pause;

predictYbTrain = forwardPropagate(Xb_norm, ThetaB1, ThetaB2, ThetaB3, Yb);

JB = ((predictYbTrain - Yb)' * (predictYbTrain - Yb) )/ (2 * mTrainB);

% Add on the penalty for regularization
JB += (sum(sum(ThetaB1(2:end, :) .^ 2)) + sum(sum(ThetaB2(2:end, :) .^ 2)) + sum(sum(ThetaB3(2:end, :) .^ 2))) * (lambda / (2 * mTrainB))

figure(2)
scatter(1:mTrainB, Yb, "b")
hold on
scatter(1:mTrainB, predictYbTrain, "r", "x")
hold off

figure(4)
scatter(Xb(:, 3), Yb, "b")
hold on
scatter(Xb(:, 3), predictYbTrain, "r", "x")
hold off

predictYbValidate = forwardPropagate(validateXb_norm, ThetaB1, ThetaB2, ThetaB3, Yb);

JBValidate = ((predictYbValidate - validateYb)' * (predictYbValidate - validateYb) )/ (2 * mValidateB);

% Add on the penalty for regularization
JBValidate += (sum(sum(ThetaB1(2:end, :) .^ 2)) + sum(sum(ThetaB2(2:end, :) .^ 2)) + sum(sum(ThetaB3(2:end, :) .^ 2))) * (lambda / (2 * mValidateB))

% Write To CSV

predictYbTest = forwardPropagate(testXb_norm, ThetaB1, ThetaB2, ThetaB3, Yb);

csvwrite("result2_2.csv", predictYbTest);

fprintf('Done. Press Enter\n');