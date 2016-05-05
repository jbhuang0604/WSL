function [loss_list, grad_list] = general_smooth_hinge(z, alpha)

% This code implements the generalized smooth hinge loss from
% http://qwone.com/~jason/writing/smoothHinge.pdf
% written by Hyun Oh Song

% AUTORIGHTS
% ---------------------------------------------------------
% Copyright (c) 2014, Hyun Oh Song
% 
% This file is part of the Song-ICML2014 code and is available 
% under the terms of the Simplified BSD License provided in 
% LICENSE. Please retain this notice and LICENSE if you use 
% this file (or any portion of it) in your project.
% ---------------------------------------------------------

if nargin < 2
    alpha = 100;
end

loss_list = zeros(length(z), 1);

% vectorized version
negs  = find(z <= 0);
loss_list(negs) = alpha/(alpha+1) - z(negs);

supps = find(z >0 & z <1);
loss_list(supps) = 1/(alpha+1)* z(supps).^(alpha+1) - z(supps) + alpha/(alpha+1);

% compute gradient
if nargout == 2
    grad_list = zeros(length(z), 1);

    % vectorized version
    grad_list(negs)   = -1;
    grad_list(supps)  = z(supps).^alpha - 1;
else
    grad_list = nan;
end
