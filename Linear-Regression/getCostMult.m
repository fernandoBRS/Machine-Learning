%	Compute cost for linear regression with multiple variables
%   J = COMPUTECOSTMULTI(X, y, theta) computes the cost of using theta as the
%   parameter for linear regression to fit the data points in X and y

function J = computeCostMulti(X, y, theta)
	m = length(y); % number of training examples
	J = (X * theta - y)' * (X * theta - y) / (2 * m);
end
