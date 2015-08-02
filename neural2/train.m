function [TrainedTheta1, TrainedTheta2, TrainedTheta3] = train(X, y, Theta1, Theta2, Theta3, alpha, mu, lambda, iters, plotIters, max)

m = size(X, 1);
JAll = zeros(iters / plotIters, 1);

TrainedTheta1 = Theta1;
TrainedTheta2 = Theta2;
TrainedTheta3 = Theta3;
Y = repmat(y, 1, size(Theta3, 2));

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
        J = sum(sum((a4 - Y) .^ 2))/ (2 * m)

        % Add on the penalty for regularization
        %J += (sum(sum(TrainedTheta1(2:end, :) .^ 2)) + sum(sum(TrainedTheta2(2:end, :) .^ 2)) + sum(sum(TrainedTheta3(2:end, :) .^ 2))) * (lambda / (2 * m))
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
    Theta3_grad = Delta3' / m + (lambda/m) * [zeros(size(TrainedTheta3, 1), 1), TrainedTheta3(:, 2:end)];
    Theta2_grad = Delta2' / m + (lambda/m) * [zeros(size(TrainedTheta2, 1), 1), TrainedTheta2(:, 2:end)];
    Theta1_grad = Delta1' / m + (lambda/m) * [zeros(size(TrainedTheta1, 1), 1), TrainedTheta1(:, 2:end)];

    %TrainedTheta1 = TrainedTheta1 - alpha * Theta1_grad;
    %TrainedTheta2 = TrainedTheta2 - alpha * Theta2_grad;
    TrainedTheta3 = TrainedTheta3 - alpha * Theta3_grad;

    % The derivative with respect to the output units is 1 so we simply elemtwise square the output of the previous layer to get the hessian (and divide by m).
    hessdelta4 = ones(m, 1);
    hessDelta3 = hessdelta4' * (a3 .^ 2) / m; 

    % Backpropagate. This has been checked by gradient checking and is correct
    hessdelta3 = (sigmoidGradient(z3) .^ 2) .* (hessdelta4 * (TrainedTheta3 .^ 2)')(:, 2:end) - sigmoidGradient2(z3) .* ((-delta4) * TrainedTheta3')(:, 2:end);
    hessDelta2 = hessdelta3' * (a2 .^ 2) / m;

    hessdelta2 = (sigmoidGradient(z2) .^ 2) .* (hessdelta3 * (TrainedTheta2 .^ 2)')(:, 2:end) - sigmoidGradient2(z2) .* ((-delta3) * TrainedTheta2')(:, 2:end);
    hessDelta1 = hessdelta2' * (X .^ 2) / m;

    % Newtonian Step, converges to 9001.1 always for some unknown reason
    TrainedTheta1 = TrainedTheta1 - Theta1_grad ./ (abs(hessDelta1)' + mu);
    TrainedTheta2 = TrainedTheta2 - Theta2_grad ./ (abs(hessDelta2)' + mu);
    %TrainedTheta3 = TrainedTheta3 - Theta3_grad ./ (abs(hessDelta3)' + mu);

end
figure(10)
plot(JAll)
end
