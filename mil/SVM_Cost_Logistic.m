function [cost, grad] = SVM_Cost_Logistic(w, X, y, C)

% work with dense sparse matrix X and compute [cost,grad] for
% Logistic loss SVM

% INPUT
% X: num_data * feat_dim
% y: num_data * 1

% OUTPUT
% cost: scalar
% grad: feat_dim * 1

num_data = size(X,1);
feat_dim = size(X,2);

scores = X*w;

% cost
s = -1* y .* scores;
loss = zeros(num_data,1);
loss(s<0)  = log(1 + exp(s(s<0)));
loss(s>=0) = s(s>=0) + log(1 + exp(-s(s>=0)));
cost = 0.5*norm(w)^2 + C * sum(loss);

% gradient
% loss_grad = -1* y ./ (1 + exp(y .* scores));
% loss_grad = bsxfun(@times, X', loss_grad');
% loss_grad = sum(loss_grad,2);

loss_grad_mat = ( (-1* y ./ (1 + exp(y .* scores)))' * X)';

grad = w + C * loss_grad_mat;
