function [theta, J_history] = gradientDescentMulti(X, y, theta, alpha, lambda, num_iters)
%GRADIENTDESCENTMULTI Performs gradient descent to learn theta
%   theta = GRADIENTDESCENTMULTI(x, y, theta, alpha, num_iters) updates theta by
%   taking num_iters gradient steps with learning rate alpha

% Initialize some useful values
m = length(y); % number of training examples
J_history = zeros(num_iters, 1);
%X_norm = featureNormalize(X);

for iter = 1:num_iters

    % ====================== YOUR CODE HERE ======================
    % Instructions: Perform a single gradient step on the parameter vector
    %               theta. 
    %
    % Hint: While debugging, it can be useful to print out the values
    %       of the cost function (computeCostMulti) and gradient here.
    %

%theta = theta - alpha * (((X * theta - y)' * X)/m)'
printf(num2str(i));
theta(1) = theta(1) - alpha * (((X*theta - y)' * X)/m)(1)'
theta(2:end) = theta(2:end) - alpha * ((((X*theta - y)' * X)(2:end) + lambda * theta(2:end)')/m)'
%theta = theta - alpha * (((X_norm * theta - y)' * X_norm)/m)'






    % ============================================================

    % Save the cost J in every iteration    
    %J_history(iter) = computeCostMulti(X_norm, y, theta)
    J_history(iter) = computeCostMulti(X, y, theta);
end

end
