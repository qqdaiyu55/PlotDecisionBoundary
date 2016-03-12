function [J, grad] = costFunction(theta, X, y)
%COSTFUNCTION Compute cost and gradient for logistic regression
%   J = COSTFUNCTION(theta, X, y) computes the cost of using theta as the
%   parameter for logistic regression and the gradient of the cost
%   w.r.t. to the parameters.

m = length(y); % number of training examples

h = sigmoid(X * theta);
cost = -y .* log(h) - (1 - y) .* log(1 - h);
J = (1 / m) * sum(cost);
grad = (1 / m) .* (X' * (h - y));


% Cost function with regularization
% h = sigmoid(X * theta);
% cost = -y .* log(h) - (1 - y) .* log(1 - h);
% thetaExcludingZero = [ [ 0 ]; theta([2:length(theta)]) ];
% J = (1 / m) * sum(cost) + (lambda / (2 * m)) * sum(thetaExcludingZero .^ 2);
% grad = (1 / m) .* (X' * (h - y)) + (lambda / m) * thetaExcludingZero;

% =============================================================

end
