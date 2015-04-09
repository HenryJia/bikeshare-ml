function [result, theta] = octave(plotX, X, y, alpha, lambda, iter)

%octave(X1, X1_norm, Y, 0.2, 0, 50)

%load("-ascii", "trainP5.csv" );

%X = trainP2(:, 1:8);

%y = trainP2(:, end);

if X(:,1) != ones(size(X, 1) ,1)
    X = [:,  ones(size(X, 1) ,1), X];
end

[theta, cost] = gradientDescentMulti(X, y, ones(size(X, 2), 1), alpha, lambda, iter)

m = length(y);

%theta = normalEqn(X, y)

cost = ((X * theta - y)' * (X * theta - y) )/ (2 * m)

figure(1)

scatter(plotX(:, 4), y)
%scatter((1:m) ,y)

hold on

plot(plotX(:, 4), X * theta, "r")
%plot(X * theta, "r")

hold off

figure(2)

%scatter(plotX(:, 4), y)
scatter((1:m) ,y)

hold on

%plot(plotX(:, 4), X * theta, "r")
plot(X * theta, "r")

hold off

%figure(3)

%plot(cost)

%hold on

result = X * theta;

%Write the thetas to csv
%{
testX = load("-ascii", "testPF1.csv" );

testX_norm = featureNormalize(testX);

if testX_norm(:,1) != ones(size(testX_norm, 1) ,1)
    testX_norm = [:,  ones(size(testX_norm, 1) ,1), testX_norm];
end

resultTest = testX_norm * theta;

csvwrite("resultTest2.csv", resultTest);
%}
end