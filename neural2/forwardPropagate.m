function PredictY = forwardPropagate(X, Theta1, Theta2, TrainY)

z2 = X * Theta1;
a2 = sigmoid(z2);

a2 = [ones(length(a2), 1), a2];
z3 = a2 * Theta2;
a3 = sigmoid(z3);

a3 = mean(a3, 2);

PredictY = ((a3 - 0.5) / 0.5) * range(TrainY) + min(TrainY);
for i = 1:size(PredictY, 1)
    if(PredictY(i) < min(TrainY))
        PredictY(i) = min(TrainY);
    end
end
end