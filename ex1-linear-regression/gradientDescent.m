function [theta, J_history] = gradientDescent(X, y, theta, alpha, num_iters)

% The values were given to us by the programmer.
% num_iters = iterations = 1500; alpha = 0.01;

%-----------------------------------------------------------
% Our gradient descent formula
% ?_j := ?_j - alpha * 1/m * [?{i=1 to m} (h?(x^(i)) - y^(i))*x^(i)]
% Our aim is minimize the ?_0 and ?_1
% This algorithm is based on updating the parameters () by taking the partial derivative of the cost function.
% Thus, the cost function is minimized and theta parameters are found for the line that best represents the samples.
%-----------------------------------------------------------

% Initialize some useful values
m = length(y); % number of training examples
J_history = zeros(num_iters, 1); 

for iter = 1:num_iters
    
    % X is mx2 matrix and theta is 2x1 matrix. The result of this product will be an mx1  matrix.
    h_theta = X * theta;
    
    % Vector operations are performed to all members of the vector before the target is updated.
    theta = theta - (alpha/m) * (X' * (h_theta - y));

    % Save the cost J in every iteration    
    J_history(iter) = computeCost(X, y, theta);

end

end
