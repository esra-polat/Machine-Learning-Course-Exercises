function [X_poly] = polyFeatures(X, p)

X_poly = zeros(numel(X), p);

X_poly(:,1:p) = X(:,1) .^ (1:p);

% for i = 1:p
%     X_poly(:,i) = X(:,1).^i;
% end

end
