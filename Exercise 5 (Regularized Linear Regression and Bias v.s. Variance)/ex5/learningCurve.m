function [error_train, error_val] = learningCurve(X, y, Xval, yval, lambda)

m = size(X, 1);

error_train = zeros(m, 1);
error_val   = zeros(m, 1);

for i = 1:m
      X_train = X(1:i,:);
      y_train = y(1:i);
      theta = trainLinearReg(X_train, y_train, lambda);
      error_train(i) = linearRegCostFunction(X_train, y_train, theta, 0); % for lambda = 0;
      error_val(i)   = linearRegCostFunction(Xval, yval, theta, 0); % for lambda = 0;
end

end
