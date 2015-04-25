function PredictY = forwardPropagate(X, Theta1, Theta2, Theta3, TrainY)

z2 = X * Theta1;
a2 = sigmoid(z2);

a2 = [ones(length(a2), 1), a2];
z3 = a2 * Theta2;
a3 = sigmoid(z3);

a3 = [ones(length(a3), 1), a3];
z4 = a3 * Theta3;
a4 = z4;

PredictY = mean(a4, 2);
for i = 1:length(PredictY)
    if(PredictY(i) < min(TrainY))
        PredictY(i) = min(TrainY);
    end
end
end