%	Predict whether the label is 0 or 1 using learned logistic regression parameters theta
%   p = PREDICT(theta, X) computes the predictions for X using a 
%   threshold at 0.5 (i.e., if sigmoid(theta'*x) >= 0.5, predict 1)

function p = predict(theta, X)
	p = sigmoid(X * theta) >= 0.5;
end
