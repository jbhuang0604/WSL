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

function cache_fc7_features(imdb, varargin)

ip = inputParser;
ip.addRequired('imdb', @isstruct);
ip.addOptional('start', 1, @isscalar);
ip.addOptional('end', 0, @isscalar);
ip.addOptional('crop_mode', 'warp', @isstr);
ip.addOptional('crop_padding', 16, @isscalar);
ip.addOptional('net_file', '', @isstr);
ip.addOptional('cache_name', '', @isstr);

ip.parse(imdb, varargin{:});
opts = ip.Results;
opts.net_def_file = './prototxt/caffenet_fc7.prototxt';
opts.batch_size = 50;
opts.crop_size = 227;

image_ids = imdb.image_ids;
if opts.end == 0
  opts.end = length(image_ids);
end

% Where to save feature cache
if ~exist('cache','file')
    mkdir('cache');
end
opts.train_output_dir = ['./cache/' opts.cache_name '/mil_train/'];
mkdir_if_missing(opts.train_output_dir);
opts.test_output_dir = ['./cache/' opts.cache_name '/mil_test/'];
mkdir_if_missing(opts.test_output_dir);

fprintf('\n\n~~~~~~~~~~~~~~~~~~~~~~~~~~~~\n');
fprintf('Feature caching options:\n');
disp(opts);
fprintf('~~~~~~~~~~~~~~~~~~~~~~~~~~~~\n\n');

% load the region of interest database
roidb = imdb.roidb_func(imdb);

caffe('init', opts.net_def_file, opts.net_file, 'test');

% caffe('set_mode_cpu');
caffe('set_mode_gpu');
% caffe('set_device',3);

total_time = 0;
count = 0;

load('results_maskout_regions.mat');
for i = opts.start:opts.end
  fprintf('%s: cache features: %d/%d\n', procid(), i, opts.end);
  
  % using all the proposals for mil testing
  test_save_file = [opts.test_output_dir image_ids{i} '.mat'];
  if exist(test_save_file, 'file') ~= 0
    fprintf(' [already exists]\n');
    continue;
  end
  count = count + 1;
  tot_th = tic;
  d = roidb.rois(i);
  im = imread(imdb.image_at(i));
  if size(im,3)~=3
      im = cat(3,im,im,im);
  end
  th = tic;
  d.feat = cache_bb_features(im, d.boxes, opts, 1);
  fprintf(' [features: %.3fs]\n', toc(th));
  th = tic;
  save(test_save_file, '-struct', 'd');
  fprintf(' [saving:   %.3fs]\n', toc(th));
  total_time = total_time + toc(tot_th);
  fprintf(' [avg time: %.3fs (total: %.3fs)]\n', ...
      total_time/count, total_time);

  % using the selected proposals for mil training
  train_save_file = [opts.train_output_dir image_ids{i} '.mat'];
  if exist(train_save_file, 'file') ~= 0
    fprintf(' [already exists]\n');
    continue;
  end
  num_gt = sum(d.gt);
  IND_GT = find(d.gt == 1);
  ind = [IND_GT;IND{i}+num_gt];
  ind = unique(ind);
  d.boxes = d.boxes(ind,:);
  d.class = d.class(ind,:);
  d.gt = d.gt(ind,:);
  d.overlap = d.overlap(ind,:);
  d.feat = d.feat(ind,:);
  save(train_save_file, '-struct', 'd');
end
