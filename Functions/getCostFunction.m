
% Cost function is a technique that will help to figure out how to fit
% the best possible straight line to a given data.

% Arguments:
% X = Matrix containing training examples.
% y = Matrix containing class labels.
% theta = Matrix containing the values [theta0, theta1].

% Output:
% J(theta).

function J = getCostFunction(X, y, theta)
	m = size(X, 1);						% Number of training examples (number of lines).
	predictions = X*theta;				% Predictions of hypothesis on all m examples.
	sqrErrors = (predictions - y).^2;	% Squared errors
	J = 1/(2*m) * sum(sqrErrors);
