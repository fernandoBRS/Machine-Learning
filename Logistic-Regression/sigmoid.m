%	Compute sigmoid functoon
%   J = SIGMOID(z) computes the sigmoid of z.

function g = sigmoid(z)
	g = 1 ./ (ones(size(z)) + exp(-z));
end
