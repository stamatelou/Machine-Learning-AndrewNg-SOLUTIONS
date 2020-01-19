function [X_poly] = polyFeatures(X, p)
%POLYFEATURES Maps X (1D vector) into the p-th power
%   [X_poly] = POLYFEATURES(X, p) takes a data matrix X (size m x 1) and
%   maps each example into its polynomial features where
%   
%


% You need to return the following variables correctly.
X_poly = zeros(numel(X), p);
m = size(X, 1);

for i = 1:m
  n = 1;
  while (n <= p)
  add = X(i).^n;
  X_poly(i,n) = add;
  n = n +1 ;
  endwhile

endfor

% ====================== YOUR CODE HERE ======================
% Instructions: Given a vector X, return a matrix X_poly where the p-th 
%               column of X contains the values of X to the p-th power.
%
% 






% =========================================================================

end
