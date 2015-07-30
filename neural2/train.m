function [TrainedTheta1, TrainedTheta2, TrainedTheta3] = train(X, Y, Theta1, Theta2, Theta3, alpha, lambda, iters, plotIters, max)

m = size(X, 1);
JAll = zeros(iters / plotIters, 1);

TrainedTheta1 = Theta1;
TrainedTheta2 = Theta2;
TrainedTheta3 = Theta3;
Y = repmat(Y, 1, size(Theta3, 2));

for i = 1:iters

    % Forward Propagation
    z2 = X * TrainedTheta1;
    a2 = sigmoid(z2);

    a2 = [ones(length(a2), 1), a2];
    z3 = a2 * TrainedTheta2;
    a3 = sigmoid(z3);
    
    a3 = [ones(length(a3), 1), a3];
    z4 = a3 * TrainedTheta3;
    a4 = z4;

    % For Debugging The Learning Algorithm Only
        
    if(fmod(i, plotIters) == 0)
        % Cost Function
        J = sum(sum((a4 - Y) .^ 2))/ (2 * m);

        % Add on the penalty for regularization
        J += (sum(sum(TrainedTheta1(2:end, :) .^ 2)) + sum(sum(TrainedTheta2(2:end, :) .^ 2)) + sum(sum(TrainedTheta3(2:end, :) .^ 2))) * (lambda / (2 * m))
        JAll(i / plotIters) = J;
    end
    

    % Calculate small delta
    delta4 = a4 - Y;

    delta3 = (delta4 * TrainedTheta3')(:, 2:end)  .* sigmoidGradient(z3);

    delta2 = (delta3 * TrainedTheta2')(:, 2:end)  .* sigmoidGradient(z2);

    % Accumulate small delta to calculate big delta which is the partial derivatives
    Delta3 = delta4' * a3;
    Delta2 = delta3' * a2;
    Delta1 = delta2' * X;

    % Finish off the calculation and add on the penalty term for regularization
    Theta3_grad = Delta3' ./ m + (lambda/m) * [zeros(size(TrainedTheta3, 1), 1), TrainedTheta3(:, 2:end)];
    Theta2_grad = Delta2' ./ m + (lambda/m) * [zeros(size(TrainedTheta2, 1), 1), TrainedTheta2(:, 2:end)];
    Theta1_grad = Delta1' ./ m + (lambda/m) * [zeros(size(TrainedTheta1, 1), 1), TrainedTheta1(:, 2:end)];

    %TrainedTheta1 = TrainedTheta1 - alpha * Theta1_grad;
    %TrainedTheta2 = TrainedTheta2 - alpha * Theta2_grad;
    %TrainedTheta3 = TrainedTheta3 - alpha * Theta3_grad;

    %The derivative with respect to the output units is 1 so we simply elemtwise square the output of the previous layer to get the hessian (and divide by m).
    epsilon = 10 ^ (-5);
    hessdelta4 = ones(m, 1);
    %hessDelta4 = sum(a3 .^ 2) / m;
    hessDelta4 = hessdelta4' * (a3 .^ 2) / m; 
    %{
    testhess1 = ((a4 - Y)' * a3 / m);

    TrainedTheta3(2,1) = TrainedTheta3(2,1) + epsilon;

    z2 = X * TrainedTheta1;
    a2 = sigmoid(z2);

    a2 = [ones(length(a2), 1), a2];
    z3 = a2 * TrainedTheta2;
    a3 = sigmoid(z3);
    
    a3 = [ones(length(a3), 1), a3];
    z4 = a3 * TrainedTheta3;
    a4 = z4;

    testhess2 = ((a4 - Y)' * a3 / m);

    ((testhess2 - testhess1) / epsilon)(1,1:5)
    hessDelta4(1,1:5)

    TrainedTheta3(2,1) = TrainedTheta3(2,1) - epsilon;
    %}
    % For the 3rd layer
    hessdelta3 = (sigmoidGradient(z3) .^ 2) .* (hessdelta4 * (TrainedTheta3 .^ 2)')(:, 2:end) + sigmoidGradient2(z3) .* (delta4 * TrainedTheta3')(:, 2:end);

    hessDelta3 = hessdelta3' * (a2 .^ 2) / m;

    testhess1 = (((a4 - Y)  * TrainedTheta3')(:, 2:end)  .* sigmoidGradient(z3))' * a2 / m;

    TrainedTheta2(2,1) = TrainedTheta2(2,1) + epsilon;

    z2 = X * TrainedTheta1;
    a2 = sigmoid(z2);

    a2 = [ones(length(a2), 1), a2];
    z3 = a2 * TrainedTheta2;
    a3 = sigmoid(z3);
    
    a3 = [ones(length(a3), 1), a3];
    z4 = a3 * TrainedTheta3;
    a4 = z4;

    testhess2 = (((a4 - Y)  * TrainedTheta3')(:, 2:end)  .* sigmoidGradient(z3))' * a2 / m;

    ((testhess2 - testhess1) / epsilon)(1,1:5)
    hessDelta3(1,1:5)

end
figure(10)
plot(JAll)
end
