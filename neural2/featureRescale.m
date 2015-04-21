function norm = featureRescale(X)

norm = zeros(size(X));

for i = 1:size(X, 2)
    norm(:, i) = (X(:, i) - min(X(:, i))) / range(X(:, i));
end

end