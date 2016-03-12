%% ================== Initialization =================
clear ; close all; clc

data = load('data.txt');
X = data(:, 1:2);
y = data(:, 3);
[m, n] = size(X);

%% =================== Gradient Descent ===============
% Features normalized
mu = mean(X);
sigma = std(X);
X_norm = (X - repmat(mu, length(X), 1)) ./ repmat(sigma, length(X), 1);
X_norm = [ones(m, 1) X_norm];

% Choose some alpha value
alpha = 0.01;
num_iters = 400;

% Init Theta and Run Gradient Descent
% Initial theta is quite important, you could change it and see the result
% theta = zeros(n + 1, 1);
theta = [-20;0.1;0.1];
[theta, J_history] = gradientDescentMulti(X_norm, y, theta, alpha, num_iters);

% Plot Boundary
plotDecisionBoundary(theta, X_norm, y);

fprintf('\nProgram paused. Press enter to continue.\n');
pause;
%% =================== Normal Equation ================
X = [ones(m, 1) X];
theta = normalEqn(X, y);

% Plot Boundary
plotDecisionBoundary(theta, X, y);