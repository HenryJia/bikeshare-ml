function [TrainedTheta1, TrainedTheta2] = train(X, Y, Theta1, Theta2, alpha, lambda, iters, plotIters, max)

m = size(X, 1);
J = 0;
JAll = zeros(iters / plotIters, 1);

TrainedTheta1 = Theta1;
TrainedTheta2 = Theta2;

%count = 1;

for i = 1:iters

    % Forward Propagation
    z2 = X * TrainedTheta1;
    a2 = log(z2);

    a2 = [ones(length(a2), 1), a2];
    z3 = a2 * TrainedTheta2;
    a3 = log(z3);

    % For Debugging The Learning Algorithm Only (Line 10 And 56 As Well)
    %{    
    if(fmod(i, plotIters) == 0)
        % Cost Function
        J = ((a3 - Y)' * (a3 - Y) )/ (2 * m);

        % Add on the penalty for regularization
        J += (sum(sum(Theta1(2:end, :) .^ 2)) + sum(sum(Theta2(2:end, :) .^ 2))) * (lambda / (2 * m));
        JAll(count / plotIters) = J;
        %plot(JAll)
        %usleep(1)
    end
    %}
    
    % Calculate small delta
    delta3 = a3 - Y;
    delta2 = (delta3 * Theta2')(:, 2:end)  .* (1 ./ z2);
    ds3 = size(delta3);
    ds2 = size(delta2);
    % Accumulate small delta to calculate big delta which is the partial derivatives
    Delta2 = delta3' * a2;
    Delta1 = delta2' * X;

    Ds2 = size(Delta2);
    Ds1 = size(Delta1);
    % Finish off the calculation and add on the penalty term for regularization
    Theta2_grad = Delta2' ./ m + (lambda/m) * [zeros(size(Theta2, 1), 1), Theta2(:, 2:end)];
    Theta1_grad = Delta1' ./ m + (lambda/m) * [zeros(size(Theta1, 1), 1), Theta1(:, 2:end)];

    TrainedTheta1 = TrainedTheta1 - alpha * Theta1_grad;
    TrainedTheta2 = TrainedTheta2 - alpha * Theta2_grad;
    
    %count += 1;
end
%plot(JAll)
end