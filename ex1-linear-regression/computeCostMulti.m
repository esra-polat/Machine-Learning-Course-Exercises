function J = computeCostMulti(X, y, theta)

% Initialize some useful values
m = length(y); % number of training examples

J = 0;

h_theta  = X * theta;
sq_error = (h_theta - y) .^ 2;    
J = (1 / (2*m)) * sum(sq_error);

% J = (1/(2*m)) * (sum(((X*theta)-y).^2));

end
