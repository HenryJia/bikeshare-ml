function PredictY = forwardPropagate(X, Theta1, Theta2, TrainY)

z2 = X * Theta1;
a2 = log(z2);

a2 = [ones(length(a2), 1), a2];
z3 = a2 * Theta2;
a3 = log(z3);

PredictY = a3 * range(TrainY) + min(TrainY);
end