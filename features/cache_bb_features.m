% AUTORIGHTS
% ---------------------------------------------------------
% Copyright (c) 2014, Ross Girshick
% Copyright (c) 2016, Dong Li
% 
% This file is part of the WSL code and is available 
% under the terms of the MIT License provided in 
% LICENSE. Please retain this notice and LICENSE if you use 
% this file (or any portion of it) in your project.
% ---------------------------------------------------------

function feat = cache_bb_features(im, boxes, opts, flag)

[batches, batch_padding] = extract_regions(im, boxes, opts, flag);
batch_size = opts.batch_size;

% compute features for each batch of region images
feat_dim = -1;
feat = [];
curr = 1;
for j = 1:length(batches)
  % forward propagate batch of region images 
  f = caffe('forward', batches(j));
  f = f{1};
  f = f(:);
  
  % first batch, init feat_dim and feat
  if j == 1
    feat_dim = length(f)/batch_size;
    feat = zeros(size(boxes,1), feat_dim, 'single');
  end

  f = reshape(f, [feat_dim batch_size]);

  % last batch, trim f to size
  if j == length(batches)
    if batch_padding > 0
      f = f(:, 1:end-batch_padding);
    end
  end

  feat(curr:curr+size(f,2)-1,:) = f';
  curr = curr + batch_size;
end
