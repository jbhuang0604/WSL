% AUTORIGHTS
% ---------------------------------------------------------
% Copyright (c) 2016, Dong Li
% 
% This file is part of the WSL code and is available 
% under the terms of the MIT License provided in 
% LICENSE. Please retain this notice and LICENSE if you use 
% this file (or any portion of it) in your project.
% ---------------------------------------------------------

function im = preprocess(im)

im = imresize(im, [227 227], 'bilinear');
% permute from RGB to BGR and subtract the data mean
im = im(:,:,[3 2 1]); % RGB to BGR
im(:,:,1) = im(:,:,1) - 104; % subtract B mean 104
im(:,:,2) = im(:,:,2) - 117; % subtract G mean 117
im(:,:,3) = im(:,:,3) - 123; % subtract R mean 123
% make width the fastest dimension, convert to single
im = permute(im, [2 1 3]);
