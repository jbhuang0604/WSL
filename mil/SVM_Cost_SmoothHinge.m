function [cost, grad] = SVM_Cost_SmoothHinge(w, X, y, C, sharpness)

% work with dense sparse matrix X and compute [cost,grad] for
% L1hinge loss SVM

% INPUT
% X: num_data * feat_dim
% y: num_data * 1

% OUTPUT
% cost: scalar
% grad: feat_dim * 1

if nargin < 5
    sharpness = 100;
end

scores = X*w;

% cost
margins = y .* scores;
[loss, loss_grad] = general_smooth_hinge(margins, sharpness);
cost = 0.5*norm(w)^2 + C * sum(loss);

% gradient
%loss_grad = loss_grad .* bag_labels;
%loss_grad = bsxfun(@times, X', loss_grad');

loss_grad_mat = ((loss_grad .* y)' * X)';

grad = w + C * loss_grad_mat;
