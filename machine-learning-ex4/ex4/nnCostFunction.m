function [J grad] = nnCostFunction(nn_params, ...
                                   input_layer_size, ...
                                   hidden_layer_size, ...
                                   num_labels, ...
                                   X, y, lambda)
%NNCOSTFUNCTION Implements the neural network cost function for a two layer
%neural network which performs classification
%   [J grad] = NNCOSTFUNCTON(nn_params, hidden_layer_size, num_labels, ...
%   X, y, lambda) computes the cost and gradient of the neural network. The
%   parameters for the neural network are "unrolled" into the vector
%   nn_params and need to be converted back into the weight matrices. 
% 
%   The returned parameter grad should be a "unrolled" vector of the
%   partial derivatives of the neural network.
%

% Reshape nn_params back into the parameters Theta1 and Theta2, the weight matrices
% for our 2 layer neural network
Theta1 = reshape(nn_params(1:hidden_layer_size * (input_layer_size + 1)), ...
                 hidden_layer_size, (input_layer_size + 1));

Theta2 = reshape(nn_params((1 + (hidden_layer_size * (input_layer_size + 1))):end), ...
                 num_labels, (hidden_layer_size + 1));

% Setup some useful variables
m = size(X, 1);
         
% You need to return the following variables correctly 
J = 0;
Theta1_grad = zeros(size(Theta1));
Theta2_grad = zeros(size(Theta2));

X = [ones(m, 1) X];
z2 = X * Theta1';
g2 = sigmoid(z2);

g2 = [ones(m, 1) g2];
z3 = g2 * Theta2';
g3 = sigmoid(z3);

eye_matrix = eye(num_labels);
y_matrix = eye_matrix(y,:);

basic_test = zeros(m,num_labels);
J = 1/m;

first_part = -(log(g3).*y_matrix);
second_part = (log(1 - g3)).*(1-y_matrix);
sm=sum(sum(first_part-second_part));
   
J = J*sm;

reg1 = lambda/(2*m);
th1 = Theta1(:,2:end);  
th2 = Theta2(:,2:end); 
reg2 = sum(sum(th1.^2)) + sum(sum(th2.^2));
 
reg = reg1 * reg2;
J = J + reg;

# gradient 
d3 = g3 - y_matrix ;

sg2 = sigmoidGradient(z2);
d2 = (d3 * th2).*sg2; 

Delta2 = d3' * g2;
Delta1 = d2' * X;

sample_size = 1/m;
reg_sample_size =lambda/m;
Theta1_grad = sample_size * Delta1 ;
#Theta1(:,1) = 0;
#Theta1_grad = Theta1_grad + reg_sample_size * Theta1.^2;
Theta2_grad = sample_size * Delta2;
#Theta2(:,1) = 0;
#Theta2_grad = Theta2_grad + reg_sample_size * Theta2.^2;


Theta1(:,1) = 0;% set the 1st column of all rows to 0
Theta1 = Theta1.*(lambda/m);
Theta1_grad = Theta1_grad + Theta1;
Theta2(:,1) = 0;
#Theta2 = [zeros(size(Theta2,1),1),Theta2];
Theta2 = Theta2.*(lambda/m);

Theta2_grad = Theta2_grad + Theta2;
% ====================== YOUR CODE HERE ======================
% Instructions: You should complete the code by working through the
%               following parts.
%
% Part 1: Feedforward the neural network and return the cost in the
%         variable J. After implementing Part 1, you can verify that your
%         cost function computation is correct by verifying the cost
%         computed in ex4.m
%
% Part 2: Implement the backpropagation algorithm to compute the gradients
%         Theta1_grad and Theta2_grad. You should return the partial derivatives of
%         the cost function with respect to Theta1 and Theta2 in Theta1_grad and
%         Theta2_grad, respectively. After implementing Part 2, you can check
%         that your implementation is correct by running checkNNGradients
%
%         Note: The vector y passed into the function is a vector of labels
%               containing values from 1..K. You need to map this vector into a 
%               binary vector of 1's and 0's to be used with the neural network
%               cost function.
%
%         Hint: We recommend implementing backpropagation using a for-loop
%               over the training examples if you are implementing it for the 
%               first time.
%
% Part 3: Implement regularization with the cost function and gradients.
%
%         Hint: You can implement this around the code for
%               backpropagation. That is, you can compute the gradients for
%               the regularization separately and then add them to Theta1_grad
%               and Theta2_grad from Part 2.
%



















% -------------------------------------------------------------

% =========================================================================

% Unroll gradients
grad = [Theta1_grad(:) ; Theta2_grad(:)];


end
