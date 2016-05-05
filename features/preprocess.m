function im = preprocess(im)

im = imresize(im, [227 227], 'bilinear');
% permute from RGB to BGR and subtract the data mean
im = im(:,:,[3 2 1]); % RGB to BGR
im(:,:,1) = im(:,:,1) - 104; % subtract B mean 104
im(:,:,2) = im(:,:,2) - 117; % subtract G mean 117
im(:,:,3) = im(:,:,3) - 123; % subtract R mean 123
% make width the fastest dimension, convert to single
im = permute(im, [2 1 3]);