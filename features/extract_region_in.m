% AUTORIGHTS
% ---------------------------------------------------------
% Copyright (c) 2016, Dong Li
% 
% This file is part of the WSL code and is available 
% under the terms of the MIT License provided in 
% LICENSE. Please retain this notice and LICENSE if you use 
% this file (or any portion of it) in your project.
% ---------------------------------------------------------

function im_in = extract_region_in(im, bbox)

crop_size = 227;
padding = 16;
scale = crop_size/(crop_size - padding*2);
half_height = (bbox(4)-bbox(2)+1)/2;
half_width = (bbox(3)-bbox(1)+1)/2;
center = [bbox(1)+half_width bbox(2)+half_height];
bbox = round([center center] + [-half_width -half_height half_width half_height]*scale);
bbox(1) = max(1, bbox(1));
bbox(2) = max(1, bbox(2));
bbox(3) = min(size(im,2), bbox(3));
bbox(4) = min(size(im,1), bbox(4));
im_in = im(bbox(2):bbox(4),bbox(1):bbox(3),:);
im_in = preprocess(im_in);
