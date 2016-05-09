% AUTORIGHTS
% ---------------------------------------------------------
% Copyright (c) 2016, Dong Li
% 
% This file is part of the WSL code and is available 
% under the terms of the MIT License provided in 
% LICENSE. Please retain this notice and LICENSE if you use 
% this file (or any portion of it) in your project.
% ---------------------------------------------------------

function [batches, batch_padding] = extract_regions(im, boxes, opts, flag)
    
if flag == 1
    
    im = single(im);
    num_boxes = size(boxes, 1);
    batch_size = opts.batch_size;
    crop_size = opts.crop_size;
    num_batches = ceil(num_boxes / batch_size);
    batch_padding = batch_size - mod(num_boxes, batch_size);
    if batch_padding == batch_size
      batch_padding = 0;
    end

    batches = cell(num_batches, 1);
    for batch = 1:num_batches
      batch_start = (batch-1)*batch_size+1;
      batch_end = min(num_boxes, batch_start+batch_size-1);

      ims = zeros(crop_size, crop_size, 3, batch_size, 'single');
      for j = batch_start:batch_end
        bbox = boxes(j,:);
        ims(:,:,:,j-batch_start+1) = extract_region_in(im, bbox);
      end

      batches{batch} = ims;
    end
    
end

if flag == 2
        
    im = single(im);
    num_boxes = size(boxes, 1);
    batch_size = opts.batch_size;
    crop_size = opts.crop_size;
    num_batches = ceil(num_boxes / batch_size);
    batch_padding = batch_size - mod(num_boxes, batch_size);
    if batch_padding == batch_size
      batch_padding = 0;
    end

    batches = cell(num_batches, 1);
    for batch = 1:num_batches
      batch_start = (batch-1)*batch_size+1;
      batch_end = min(num_boxes, batch_start+batch_size-1);

      ims = zeros(crop_size, crop_size, 3, batch_size, 'single');
      for j = batch_start:batch_end
        bbox = boxes(j,:);
        ims(:,:,:,j-batch_start+1) = extract_region_out(im, bbox);
      end

      batches{batch} = ims;
    end
    
end
