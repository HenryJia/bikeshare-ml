function [ThetaA1, ThetaA2, ThetaA3] = neural2_1

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
iters = 10;
scatterIters = 1;

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
[ThetaA1, ThetaA2, ThetaA3] = train(Xa_norm, Ya, ThetaA1, ThetaA2, ThetaA3, alpha, lambda, iters, scatterIters);
toc

fprintf('Training Complete. Calculate Costs. Press Enter\n');
pause;

predictYaTrain = forwardPropagate(Xa_norm, ThetaA1, ThetaA2, ThetaA3, Ya);

JA = ((predictYaTrain - Ya)' * (predictYaTrain - Ya) )/ (2 * mTrainA);

% Add on the penalty for regularization
JA += (sum(sum(ThetaA1(2:end, :) .^ 2)) + sum(sum(ThetaA2(2:end, :) .^ 2)) + sum(sum(ThetaA3(2:end, :) .^ 2))) * (lambda / (2 * mTrainA))

figure(1)
scatter(1:mTrainA, Ya, "b")
hold on
scatter(1:mTrainA, predictYaTrain, "r", "x")
hold off

figure(3)
scatter(Xa(:, 2), Ya, "b")
hold on
scatter(Xa(:, 2), predictYaTrain, "r", "x")
hold off

predictYaValidate = forwardPropagate(validateXa_norm, ThetaA1, ThetaA2, ThetaA3, Ya);

JAValidate = ((predictYaValidate - validateYa)' * (predictYaValidate - validateYa) )/ (2 * mValidateA);

% Add on the penalty for regularization
JAValidate += (sum(sum(ThetaA1(2:end, :) .^ 2)) + sum(sum(ThetaA2(2:end, :) .^ 2)) + sum(sum(ThetaA3(2:end, :) .^ 2))) * (lambda / (2 * mValidateA))

% Write To CSV

predictYaTest = forwardPropagate(testXa_norm, ThetaA1, ThetaA2, ThetaA3, Ya);

csvwrite("result2_1.csv", predictYaTest);

fprintf('Done. Press Enter\n');