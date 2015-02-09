%	Compute cost and gradient for logistic regression with regularization
%   J = COSTFUNCTIONREG(theta, X, y, lambda) computes the cost of using
%   theta as the parameter for regularized logistic regression and the
%   gradient of the cost w.r.t. to the parameters. 

function [J, grad] = costFunctionReg(theta, X, y, lambda)
	m = length(y); % number of training examples
	n = length(theta);
	h = sigmoid(X * theta);
	theta(1) = 0;

	J = sum((- y .* log(h) - (ones(m, 1) - y) .* log(ones(m, 1) - h))) / m + lambda / (2 * m) * sum(theta .^ 2);
	grad = (X' * (h - y)) / m + lambda / m * theta;
end
