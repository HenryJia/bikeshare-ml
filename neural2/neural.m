function neural
%(plotX, X, y, alpha, lambda, iter)

% Load and normalise data

fprintf('Load Data\n');

Xa1 = load("trainPF6_1.csv")(:, 2:end);
Xb1 = load("trainPF6_2.csv")(:, 2:end);
Ya1 = load("trainY1.csv" )(:, end);
Yb1 = load("trainY2.csv" )(:, end);
testXa1 = load("testPF3_1.csv")(:, 2:end);
testXb1 = load("testPF3_2.csv")(:, 2:end);

fprintf('Data Loaded. Normalize Features. Press Enter\n');
pause;

Xa1_norm = featureNormalize(Xa1);
Xb1_norm = featureNormalize(Xb1);
testXa1_norm = featureNormalize(testXa1);
testXb1_norm = featureNormalize(testXb1);

fprintf('Features Normalised. Gradient Descent. Press Enter\n');
pause;

% Calculate Thetas & Results For First Hidden Layer

[Xa2, thetaA1] = octave(Xa1, Xa1_norm, Ya1, 0.1, 0, 10000);
[Xb2, thetaB1] = octave(Xb1, Xb1_norm, Yb1, 0.1, 0, 10000);

fprintf('Gradient Descent Complete. Calculate Results. Press Enter\n');
pause;

% Apply Thetas From First Hdiden Layer To Test Data

if testXa1_norm(:,1) != ones(size(testXa1_norm, 1) ,1)
    testXa1_norm = [:,  ones(size(testXa1_norm, 1) ,1), testXa1_norm];
end

if testXb1_norm(:,1) != ones(size(testXb1_norm, 1) ,1)
    testXb1_norm = [:,  ones(size(testXb1_norm, 1) ,1), testXb1_norm];
end

testXa2 = testXa1_norm * thetaA1;
testXb2 = testXb1_norm * thetaB1;

fprintf('Results Calculated. Generate Data (Features). Press Enter\n');
pause;

% Generate Data (Features) For Output Layer

XaF2 = [Xa2, Xa2.^2, Xa2.^3, Xa2.^4, Xa2.^5];
XbF2 = [Xb2, Xb2.^2, Xb2.^3, Xb2.^4, Xb2.^5];
testXaF2 = [testXa2, testXa2.^2, testXa2.^3, testXa2.^4, testXa2.^5];
testXbF2 = [testXb2, testXb2.^2, testXb2.^3, testXb2.^4, testXb2.^5];

fprintf('Features Generated. Normalise Features. Press Enter\n');
pause;

% Normalise Output Of First Hidden Layer

XaF2_norm = featureNormalize(XaF2);
XbF2_norm = featureNormalize(XbF2);
testXaF2_norm = featureNormalize(testXaF2);
testXbF2_norm = featureNormalize(testXbF2);

fprintf('Features Normalised. Calculate Final Thetas. Press Enter\n');
pause;

% Calculate The Final Results

[Xa3, thetaA2] = octave(XaF2, XaF2_norm, Ya1, 0.1, 0, 10000);
[Xb3, thetaB2] = octave(XbF2, XbF2_norm, Yb1, 0.1, 0, 10000);

fprintf('Final Thetas Calculated. Calculate Final Results. Press Enter\n');
pause;

% Apply Final Thetas To The Test Data

if testXaF2_norm(:,1) != ones(size(testXaF2_norm, 1) ,1)
    testXaF2_norm = [:,  ones(size(testXaF2_norm, 1) ,1), testXaF2_norm];
end

if testXbF2_norm(:,1) != ones(size(testXbF2_norm, 1) ,1)
    testXbF2_norm = [:,  ones(size(testXbF2_norm, 1) ,1), testXbF2_norm];
end

testXa3 = testXaF2_norm * thetaA2;
testXb3 = testXbF2_norm * thetaB2;

fprintf('Calculated Final Results. Write To CSV. Press Enter\n');
pause;

% Write To CSV

csvwrite("result3_1.csv", testXa3);
csvwrite("result3_2.csv", testXb3);

fprintf('Done. Press Enter\n');
pause;