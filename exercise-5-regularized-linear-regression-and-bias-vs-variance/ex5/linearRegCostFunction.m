function [J, grad] = linearRegCostFunction(X, y, theta, lambda)

m = length(y); % number of training examples
J = 0;
grad = zeros(size(theta));

h_theta = X * theta;

J = (1/(2*m)) * sum((h_theta - y).^2) + (lambda/(2*m)) * sum(theta(2:end).^2);

grad = 1 / m * X' * (h_theta - y);
grad(2:end) += (lambda / m) * theta(2:end);

grad = grad(:);

end
