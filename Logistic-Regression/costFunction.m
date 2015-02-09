%	Compute cost and gradient for logistic regression
%   J = COSTFUNCTION(theta, X, y) computes the cost of using theta as the
%   parameter for logistic regression and the gradient of the cost
%   w.r.t. to the parameters.

function [J, grad] = costFunction(theta, X, y)
	m = length(y); % number of training examples
	h = sigmoid(X * theta);
	J = sum((- y .* log(h) - (ones(m, 1) - y) .* log(ones(m, 1) - h))) / m;
	grad = (X' * (h - y)) / m;
end
