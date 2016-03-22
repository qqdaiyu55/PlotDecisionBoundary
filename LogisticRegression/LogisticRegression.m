%% ================== Initialization ==================
clear; close all; clc

data = load('../data.txt');
X = data(:, [1, 2]); y = data(:, 3);
[m, n] = size(X);

fprintf(['Plotting data with + indicating (y = 1) examples and o ' ...
         'indicating (y = 0) examples.\n']);
plotData(X, y);

fprintf('\nProgram paused. Press enter to continue.\n');
pause;
close all;
%% ============== Optimizing using fminunc  =============
%  Set options for fminunc
options = optimset('GradObj', 'on', 'MaxIter', 400);

%  Run fminunc to obtain the optimal theta
X = [ones(m, 1) X];
initial_theta = zeros(n + 1, 1);
[theta, cost] = ...
	fminunc(@(t)(costFunction(t, X, y)), initial_theta, options);

% Plot Boundary
plotDecisionBoundary(theta, X, y);