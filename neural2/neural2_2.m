function [predictYaTrain, predictYbTrain, predictYaValidate, predictYbValidate, predictYaTest, predictYbTest] = neural2_2
%(scatterX, X, y, alpha, lambda, iter)

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
iters = 2000;
scatterIters = 40;

mTrainA = length(Ya);
mTrainB = length(Yb);

mValidateA = length(validateYa);
mValidateB = length(validateYb);

fprintf('Data Loaded. Normalised Features And Add Bias Units. Press Enter\n');
%pause;

Xa_norm =  featureNormalize(Xa);
Xb_norm =  featureNormalize(Xb);
validateXa_norm =  featureNormalize(validateXa);
validateXb_norm =  featureNormalize(validateXb);
testXa_norm =  featureNormalize(testXa);
testXb_norm =  featureNormalize(testXb);
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

ThetaA2 = abs(randInitializeWeights(22, 22));
ThetaB2 = abs(randInitializeWeights(22, 22));

ThetaA3 = abs(randInitializeWeights(22, 1));
ThetaB3 = abs(randInitializeWeights(22, 1));

% Calculate Thetas & Results For First Hidden Layer

fprintf('Thetas initialise. Training. Press Enter\n');
%pause;

%[ThetaA1, ThetaA2, ThetaA3] = train(Xa_norm, Ya_norm, ThetaA1, ThetaA2, ThetaA3, alpha, lambda, iters, scatterIters);
[ThetaB1, ThetaB2, ThetaB3] = train(Xa_norm, Ya_norm, ThetaB1, ThetaB2, ThetaB3, alpha, lambda, iters, scatterIters);

fprintf('Training Complete. Calculate Training Costs. Press Enter\n');
pause;

%predictYaTrain = forwardPropagate(Xa_norm, ThetaA1, ThetaA2, ThetaA3, Ya);
predictYbTrain = forwardPropagate(Xb_norm, ThetaB1, ThetaB2, ThetaB3, Yb);

%JA = ((predictYaTrain - Ya)' * (predictYaTrain - Ya) )/ (2 * mTrainA);
JB = ((predictYbTrain - Yb)' * (predictYbTrain - Yb) )/ (2 * mTrainB);

% Add on the penalty for regularization
%JA += (sum(sum(ThetaA1(2:end, :) .^ 2)) + sum(sum(ThetaA2(2:end, :) .^ 2)) + sum(sum(ThetaA3(2:end, :) .^ 2))) * (lambda / (2 * mTrainA))
JB += (sum(sum(ThetaB1(2:end, :) .^ 2)) + sum(sum(ThetaB2(2:end, :) .^ 2)) + sum(sum(ThetaB3(2:end, :) .^ 2))) * (lambda / (2 * mTrainB))
%{
figure(1)
scatter(1:mTrainA, Ya, "b")
hold on
scatter(1:mTrainA, predictYaTrain, "r")
hold off
%}
figure(2)
scatter(1:mTrainB, Yb, "b")
hold on
scatter(1:mTrainB, predictYbTrain, "r")
hold off
%{
figure(3)
scatter(Xa(:, 3), Ya, "b")
hold on
scatter(Xa(:, 3), predictYaTrain, "r")
hold off
%}
figure(4)
scatter(Xb(:, 3), Yb, "b")
hold on
scatter(Xb(:, 3), predictYbTrain, "r")
hold off

%predictYaValidate = forwardPropagate(validateXa_norm, ThetaA1, ThetaA2, ThetaA3, Ya);
predictYbValidate = forwardPropagate(validateXb_norm, ThetaB1, ThetaB2, ThetaB3, Yb);

%JAValidate = ((predictYaValidate - validateYa)' * (predictYaValidate - validateYa) )/ (2 * mValidateA);
JBValidate = ((predictYbValidate - validateYb)' * (predictYbValidate - validateYb) )/ (2 * mValidateB);

% Add on the penalty for regularization
%JAValidate += (sum(sum(ThetaA1(2:end, :) .^ 2)) + sum(sum(ThetaA2(2:end, :) .^ 2)) + sum(sum(ThetaA3(2:end, :) .^ 2))) * (lambda / (2 * mValidateA))
JBValidate += (sum(sum(ThetaB1(2:end, :) .^ 2)) + sum(sum(ThetaB2(2:end, :) .^ 2)) + sum(sum(ThetaB3(2:end, :) .^ 2))) * (lambda / (2 * mValidateB))

%fprintf('Calculated Final Results. Write To CSV. Press Enter\n');
%pause;

% Write To CSV

%predictYaTest = forwardPropagate(testXa_norm, ThetaA1, ThetaA2, ThetaA3, Ya);
predictYbTest = forwardPropagate(testXb_norm, ThetaB1, ThetaB2, ThetaB3. Yb);

csvwrite("result2_2.csv", predictYbTest);

fprintf('Done. Press Enter\n');
%pause;