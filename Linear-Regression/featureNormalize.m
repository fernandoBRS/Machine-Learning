%	Normalizes the features in X 
%   FEATURENORMALIZE(X) returns a normalized version of X where
%   the mean value of each feature is 0 and the standard deviation
%   is 1. This is often a good preprocessing step to do when
%   working with learning algorithms.

function [X_norm, mu, sigma] = featureNormalize(X)
	X_norm = X;
	mu = zeros(1, size(X, 2));
	sigma = zeros(1, size(X, 2));

	mu    = mean(X)
	sigma = std(X)

	indicies = 1:size(X, 2);

	for i = indicies,
	  diff  = X(:, i) - mu(i);
	  X_norm(:, i) = diff / sigma(i);
	end
end
