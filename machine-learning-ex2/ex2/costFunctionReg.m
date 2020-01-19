function [J, grad] = costFunctionReg(theta, X, y, lambda)
%COSTFUNCTIONREG Compute cost and gradient for logistic regression with regularization
%   J = COSTFUNCTIONREG(theta, X, y, lambda) computes the cost of using
%   theta as the parameter for regularized logistic regression and the
%   gradient of the cost w.r.t. to the parameters. 

% Initialize some useful values
m = length(y); % number of training examples

% You need to return the following variables correctly 
[Jfirst, gradfirst] = costFunction(theta, X, y);
reg1 = lambda/(2*m);
th = theta(2:end);
reg2 = sum(th.^2);
reg = reg1 * reg2;
J = Jfirst + reg;

k = lambda * theta(2:end) ;
c =[0;k];
grad_reg = c/m;
grad = grad_reg + gradfirst;

% ====================== YOUR CODE HERE ======================
% Instructions: Compute the cost of a particular choice of theta.
%               You should set J to the cost.
%               Compute the partial derivatives and set grad to the partial
%               derivatives of the cost w.r.t. each parameter in theta



% =============================================================

end
