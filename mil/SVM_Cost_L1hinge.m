function [cost, grad] = SVM_Cost_L1hinge(w, X, y, C)

% work with dense sparse matrix X and compute [cost,grad] for
% L1hinge loss SVM

% INPUT
% X: num_data * feat_dim
% y: num_data * 1

% OUTPUT
% cost: scalar
% grad: feat_dim * 1

scores = X*w;

% cost
margins = y .* scores;
loss    = max(0, 1 - margins);
cost    = 0.5*norm(w)^2 + C * sum(loss);

% gradient
margin_test = (margins < 1);

% loss_grad = margin_test .* (-1*y);
% loss_grad = bsxfun(@times, X', loss_grad');
% loss_grad = sum(loss_grad, 2);

loss_grad_mat = ((margin_test .* (-1*y))' * X)';

%assert(norm(loss_grad- loss_grad_mat) < 1e-7);

grad = w + C * loss_grad_mat;



