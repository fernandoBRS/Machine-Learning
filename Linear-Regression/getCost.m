%	computeCost: Function to compute cost for linear regression, using one variable.

% 	Arguments:
% 	X = Matrix containing training examples.
% 	y = Matrix containing class labels.
% 	theta = Matrix containing the values [theta0, theta1].

% 	Output:
%   J = J(theta), cost of using theta as the parameter for linear regression to fit the data points in X and y.

function J = computeCost(X, y, theta)
	m = length(y); 						% Number of training examples
	predictions = X*theta;				% Predictions of hypothesis on all m examples
	sqrErrors = (predictions - y).^2;	% Squared errors
	J = 1/(2*m)*sum(sqrErrors);
end
