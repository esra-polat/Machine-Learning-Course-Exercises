function [X_norm, mu, sigma] = featureNormalize(X)

X_norm = X;
mu = zeros(1, size(X, 2));
sigma = zeros(1, size(X, 2));

mu = mean(X); %avarage of X^(i)'s
sigma = std(X); %standard deviation of X^(i)'s

X_norm = (X-mu) ./ sigma; 

end
