function [theta, J_history] = gradientDescentMulti(X, y, theta, alpha, num_iters)
%GRADIENTDESCENTMULTI Performs gradient descent to learn theta
%   theta = GRADIENTDESCENTMULTI(x, y, theta, alpha, num_iters) updates theta by
%   taking num_iters gradient steps with learning rate alpha

% Initialize some useful values
m = length(y); % number of training examples
J_history = zeros(num_iters, 1);

for iter = 1:num_iters
    h = X * theta;
    theta = theta - alpha * (1 / m) * (X' * (h - y));

    % Save the cost J in every iteration    
    squaredErrors = (h - y) .^ 2;
    J_history(iter) = (1 / (2 * m)) * sum(squaredErrors);

end

end
