function [J grad] = nnCostFunction(nn_params, input_layer_size, hidden_layer_size, num_labels, X, y, lambda)

% for our 2 layer neural network
Theta1 = reshape(nn_params(1:hidden_layer_size * (input_layer_size + 1)), hidden_layer_size, (input_layer_size + 1));
Theta2 = reshape(nn_params((1 + (hidden_layer_size * (input_layer_size + 1))):end), num_labels, (hidden_layer_size + 1));

m = size(X, 1);
J = 0;
Theta1_grad = zeros(size(Theta1));
Theta2_grad = zeros(size(Theta2));

% Foward propagation
% a^(1) = x
% z^(2) = theta^(1) * a^(1)
% a^(2) = g(z^(2)) (add a_0^(2))
% z^(3) = theta^(2) * a^(2)
% a^(3) = h_theta(x) = g(z^(3))

a1 = [ones(m,1) X];
z2 = a1 * Theta1';  a2 = sigmoid(z2);   a2 =  [ones(size(a2,1),1) a2]; 
z3 = a2 * Theta2';  a3 = sigmoid(z3); 
h_theta_x = a3; % m x num_labels == 5000 x 10

y_vec = (1:num_labels) == y; % m x num_labels  == 5000 x 10

% Cost function without regularization
cost_term = -y_vec .* log(h_theta_x) - (1 - y_vec) .* log(1 - h_theta_x);
% Regularization to cost function
reg_term = (lambda / (2*m)) * (sum(sum(Theta1(:,2:end) .^ 2))+ sum(sum(Theta2(:, 2:end) .^ 2)));
J = (1 / m) * sum(sum(cost_term)) + reg_term;

% Back Propagation

% ==== Part 2 ====
D1 = zeros(size(Theta1));
D2 = zeros(size(Theta2));
 
for t=1:m 
    a1 = X(t,:)';
    a1 = [1; a1];
    % hidden layer
    z2 = a1' * Theta1';
    a2 = sigmoid(z2);
    a2 = [1; a2'];
    % output layer
    z3 = a2' * Theta2';
    a3 = sigmoid(z3);
 
    % compute the error
    d3 = a3 - y_vec(t, :);
    % compute error on the second layer
    tmp = (Theta2' * d3')';
    d2 = tmp(2:end).* sigmoidGradient(z2);
    D2 = D2 + d3' * a2';
    D1 = D1 + d2' * a1';
end 

Theta1_grad = (1/m) * D1;
Theta2_grad = (1/m) * D2;

% Regularized gradient
Theta2_grad(:, 2 : end) = Theta2_grad(:, 2 : end) .+ lambda * Theta2(:, 2 : end) / m;
Theta1_grad(:, 2 : end) = Theta1_grad(:, 2 : end) .+ lambda * Theta1(:, 2 : end) / m;

% Unroll gradients
grad = [Theta1_grad(:) ; Theta2_grad(:)];

end