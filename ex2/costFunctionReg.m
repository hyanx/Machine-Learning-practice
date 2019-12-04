function [J, grad] = costFunctionReg(theta, X, y, lambda)
%COSTFUNCTIONREG Compute cost and gradient for logistic regression with regularization
%   J = COSTFUNCTIONREG(theta, X, y, lambda) computes the cost of using
%   theta as the parameter for regularized logistic regression and the
%   gradient of the cost w.r.t. to the parameters. 

% Initialize some useful values
m = length(y); % number of training examples

% You need to return the following variables correctly 
J = 0;
grad = zeros(size(theta));

% ====================== YOUR CODE HERE ======================
% Instructions: Compute the cost of a particular choice of theta.
%               You should set J to the cost.
%               Compute the partial derivatives and set grad to the partial
%               derivatives of the cost w.r.t. each parameter in theta

h = sigmoid(X * theta);
grad_ = 1/m * X' *(h - y);
theta(1) = 0;
reg_grad_term = lambda / m * theta;
grad = grad_ + reg_grad_term;

regularization_term_ = lambda / (2 * m);
first_term = -y' * log(h);
second_term = (1 .- y)' * log(1 .- h);
J_ = 1/m * (first_term - second_term);
theta(1) = 0;
theta_square = theta' * theta;
regularization_term = regularization_term_ * theta_square;
J = J_ + regularization_term;

% =============================================================

end
