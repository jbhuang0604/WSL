function [cost, grad] = slslvm_cost_smoothhinge_bagmine(...
                w, pos_X, neg_X, ...
                pos_averaging_matrix, pos_cum_bag_idx, ...
                neg_averaging_matrix, neg_cum_bag_idx,...
                num_pos, num_neg, ...
                y, pweighted_y, C, mu, pweight, sharpness, bias_mult)
       
% AUTORIGHTS
% ---------------------------------------------------------
% Copyright (c) 2014, Hyun Oh Song
% 
% This file is part of the Song-ICML2014 code and is available 
% under the terms of the Simplified BSD License provided in 
% LICENSE. Please retain this notice and LICENSE if you use 
% this file (or any portion of it) in your project.
% ---------------------------------------------------------       

[f_w_pos, pos_averaging_matrix] = compute_smooth_score(...
        w, pos_cum_bag_idx, pos_averaging_matrix, pos_X, mu, bias_mult);
[f_w_neg, neg_averaging_matrix] = compute_smooth_score(...
        w, neg_cum_bag_idx, neg_averaging_matrix, neg_X, mu, bias_mult);

scores = [f_w_pos, f_w_neg]; 

% cost
pos_ind = 1 : num_pos;
neg_ind = num_pos+1 : num_pos+num_neg;

margins = y .* scores';
[loss, loss_grad] = general_smooth_hinge(margins, sharpness);
cost = 0.5*norm(w)^2 + C * ...
    (pweight * sum(loss(pos_ind)) + sum(loss(neg_ind)));
  

% gradient
yy = (loss_grad .* pweighted_y);
pos_avg_yy = pos_averaging_matrix * yy(pos_ind);
neg_avg_yy = neg_averaging_matrix * yy(neg_ind);
loss_grad_mat = pos_X * pos_avg_yy + neg_X * neg_avg_yy;
grad = w + C * [loss_grad_mat; bias_mult*(sum(pos_avg_yy) + sum(neg_avg_yy))];

cost = double(cost);
grad = double(grad);

              
function [f_w, averaging_matrix]= compute_smooth_score(...
              w, cum_bag_idx, averaging_matrix, X, mu, bias_mult)
            
num_insts = cum_bag_idx(end);
instance_scores = w(1:end-1)'*X + ...
             (w(end)*bias_mult)*ones(1,num_insts); % 1 by num_insts
           
a = zeros(num_insts, 1, 'single');

bag_start = 1;

for bag_end = cum_bag_idx
    ind = bag_start : bag_end;
     
    a( ind ) = projsplx_c_float(1/mu * instance_scores( ind ));

    bag_start = bag_end + 1;
end

% premultiply a to the averaging matrix
averaging_matrix = bsxfun(@times, averaging_matrix, a);
f_w  = (instance_scores -mu/2*a') * averaging_matrix;
