function [J, grad] = linearRegCostFunction(X, y, theta, lambda)
%LINEARREGCOSTFUNCTION Compute cost and gradient for regularized linear 
%regression with multiple variables
%   [J, grad] = LINEARREGCOSTFUNCTION(X, y, theta, lambda) computes the 
%   cost of using theta as the parameter for linear regression to fit the 
%   data points in X and y. Returns the cost in J and the gradient in grad

% Initialize some useful values
m = length(y); % number of training examples
k = 1/(2 *m);
% You need to return the following variables correctly 
J = 0;
grad = zeros(size(theta));

h = X * theta ;
d = (h-y).^2;
dsum = sum(d);
a = k * dsum ;

k1 = lambda/(2*m);
th = theta(2:end).^2;
reg = k1 * sum(th);

J = a + reg;
 

k2 = 1/m;
s = sum((h-y).*X);
a1 = k2 * s;

th = [ zeros(1, 1) ; theta(2:end) ] ;
k3 = lambda/m;
reg = k3 * th';
grad = a1 + reg;
% ====================== YOUR CODE HERE ======================
% Instructions: Compute the cost and gradient of regularized linear 
%               regression for a particular choice of theta.
%
%               You should set J to the cost and grad to the gradient.
%












% =========================================================================

grad = grad(:);

end
