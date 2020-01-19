function [J, grad] = costFunction(theta, X, y)
%COSTFUNCTION Compute cost and gradient for logistic regression
%   J = COSTFUNCTION(theta, X, y) computes the cost of using theta as the
%   parameter for logistic regression and the gradient of the cost
%   w.r.t. to the parameters.

% Initialize some useful values
m = length(y); % number of training examples
k = 1/m; 
% You need to return the following variables correctly 
z = X * theta;
h = sigmoid(z);
log1 = log(h);
log0 = log(1 - h);
a = - y' * log1;
b = (1 - y)' * log0;
J = k * (a - b);

c = h - y;
d = X' * c;
grad = k * d;

% ====================== YOUR CODE HERE ======================
% Instructions: Compute the cost of a particular choice of theta.
%               You should set J to the cost.
%               Compute the partial derivatives and set grad to the partial
%               derivatives of the cost w.r.t. each parameter in theta
%
% Note: grad should have the same dimensions as theta
%








% =============================================================

end
