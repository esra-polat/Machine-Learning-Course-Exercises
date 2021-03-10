function J = computeCost(X, y, theta)

% Initialize some useful values
m = length(y); % number of training examples

%------------------------------------------------------------------------------------
% Our cost function formula
% J(?_0, ?_1) = 1/(2m) * [?{i=1 to m} (h?(x^(i)) - y^(i))^2]
% Our aim is minimize the ?_0 and ?_1
% X values are given us, so our job will be to find the ? parameters.
% In the formula above, we found the cost function for all training set (m) examples.
% How do we find the smallest cost value?
% We can use gradientDescent to find the smallest cost value.
%------------------------------------------------------------------------------------

h_theta  = X * theta;
sq_error = (h_theta - y) .^ 2;    
J = 1 / (2*m) * sum(sq_error);

end
