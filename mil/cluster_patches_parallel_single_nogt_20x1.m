function cluster_patches_parallel_single_nogt_20x1(seed_pos_image_id, ...
                             classid, trainset, year, Q, varargin)
              
% AUTORIGHTS
% ---------------------------------------------------------
% Copyright (c) 2014, Hyun Oh Song
% Copyright (c) 2016, Dong Li
% 
% This file is part of the Song-ICML2014 code and is available 
% under the terms of the Simplified BSD License provided in 
% LICENSE. Please retain this notice and LICENSE if you use 
% this file (or any portion of it) in your project.
% ---------------------------------------------------------

if ischar(seed_pos_image_id)
  seed_pos_image_id = str2double(seed_pos_image_id); 
end         

if ischar(classid)
  classid = str2double(classid);
end

fprintf('[Processing] seed_pos_image_id: %d, classid: %d\n', ...
  seed_pos_image_id, classid);
  
if nargin <= 2
  trainset = 'trainval';
  year = '2007';
  Q = 20;
end

% Q: save only top Q% of results for space
if ischar(Q), Q = str2double(Q); end 

ip = inputParser;
%ip.addRequired('clss',     @iscell);
ip.addRequired('trainset', @isstr);
ip.addRequired('year',     @isstr);
ip.addRequired('classid',  @isscalar);

ip.addParamValue('svm_C',           10^-3,   @isscalar);
ip.addParamValue('bias_mult',       10,      @isscalar);
ip.addParamValue('pos_loss_weight', 20,       @isscalar);
ip.addParamValue('layer',           'fc7', @isstr);
ip.addParamValue('fine_tuned',      0,       @isscalar);
ip.addParamValue('use_flipped',     0,       @isscalar);

%ip.parse(clss, trainset, year, classid, jobid, num_jobs, varargin{:});
ip.parse(trainset, year, classid, varargin{:});
opts = ip.Results;

conf = voc_config();

dataset.year = year;
dataset.trainset = trainset;
dataset.image_ids = textread(sprintf(conf.pascal.VOCopts.imgsetpath, trainset), '%s');
dataset.image_ids_small = dataset.image_ids(randperm(length(dataset.image_ids), 1000));
dataset.pos_image_ids = dataset.image_ids;

% ------------------------------------------------------------------------
load('class_pos_images.mat');
pos_image_ids = class_pos_images(classid).ids;

neg_image_ids = setdiff(dataset.image_ids, pos_image_ids);

% sample_pos_image_ids = pos_image_ids(1:40);
% sample_neg_image_ids = neg_image_ids(1:200);
sample_pos_image_ids = pos_image_ids;
sample_neg_image_ids = neg_image_ids;

num_pos_images = length(sample_pos_image_ids);
num_neg_images = length(sample_neg_image_ids);
% ------------------------------------------------------------------------

% ------------------------------------------------------------------------
% Check if the seed image id exceeds the number of positive images
if seed_pos_image_id > num_pos_images
  fprintf('%d exceeds the number of pos data\n', seed_pos_image_id);
  return;
end
% ------------------------------------------------------------------------

% ------------------------------------------------------------------------
% Check if this positive image is already evaluated
if ~exist('paris_results_nogt_20x1','file')
    mkdir('paris_results_nogt_20x1');
end
save_filename = sprintf('paris_results_nogt_20x1/%s_%d.mat', ...
                       pos_image_ids{seed_pos_image_id}, classid);   
                     
if exist(save_filename, 'file') ~= 0
  fprintf('%s already computed. Exit.\n', pos_image_ids{seed_pos_image_id}); 
  return;
end
% ------------------------------------------------------------------------

% ------------------------------------------------------------------------
% For each windows in seed image, find the nearest neighbor window per all
% other pos, neg images. 
% Fill in this table (# images by #windows_in_seed_image) with tuple
% (window idx, L2 diff)

th = tic(); % measure time to process one image
%seed_pos_image_id = 10;

seed_image_struct = load_cached_features_hos(1, dataset.trainset, dataset.year, pos_image_ids{seed_pos_image_id});
% remove ground truth boxes
seed_image_windows = seed_image_struct.feat(seed_image_struct.gt~=1,:); %#data X #feat

num_seed_windows = size(seed_image_windows, 1);

% loop over positives
table_pos_idx  = uint16(zeros(num_pos_images, num_seed_windows));
table_pos_diff = single(zeros(num_pos_images, num_seed_windows));
for pos_i = 1:num_pos_images
  if mod(pos_i,100)==0, fprintf('%d/%d pos\n', pos_i, num_pos_images); end
  
  this_pos_image_windows = load_cached_features_hos(1, dataset.trainset, dataset.year, pos_image_ids{pos_i});
  % remove ground truth boxes                            
  this_pos_image_windows = this_pos_image_windows.feat(this_pos_image_windows.gt~=1,:);
  %this_pos_image_windows = double(X_pos{pos_i}.feat);
  
%  num_windows = size(this_pos_image_windows,1);
  
%   diff_ori = repmat(sum(seed_image_windows.^2,2)',num_windows,1) + ...
%          repmat(sum(this_pos_image_windows.^2,2), 1, num_seed_windows) - ...
%          2*this_pos_image_windows * seed_image_windows';

  diff = bsxfun(@plus, dot(seed_image_windows, seed_image_windows,2)', ...
            dot(this_pos_image_windows, this_pos_image_windows,2)) - ...
            2*this_pos_image_windows * seed_image_windows';
       
  [table_pos_diff(pos_i,:), table_pos_idx(pos_i,:)] = min(diff,[],1); 
end

% loop over negatives
table_neg_idx  = uint16(zeros(num_neg_images, num_seed_windows));
table_neg_diff = single(zeros(num_neg_images, num_seed_windows));
for neg_i = 1:num_neg_images
  if mod(neg_i,100)==0, fprintf('%d/%d neg\n', neg_i, num_neg_images); end
  
  this_neg_image_windows = load_cached_features_hos(1, dataset.trainset, dataset.year, neg_image_ids{neg_i});
  % remove ground truth boxes                                                    
  this_neg_image_windows = this_neg_image_windows.feat(this_neg_image_windows.gt~=1,:);
  %this_neg_image_windows = double(X_neg{neg_i}.feat);
  
%  num_windows = size(this_neg_image_windows,1);
  
%   diff_ori = repmat(sum(seed_image_windows.^2,2)',num_windows,1) + ...
%          repmat(sum(this_neg_image_windows.^2,2), 1, num_seed_windows) - ...
%          2*this_neg_image_windows * seed_image_windows';
       
  diff = bsxfun(@plus, dot(seed_image_windows, seed_image_windows,2)', ...
            dot(this_neg_image_windows, this_neg_image_windows,2)) - ...
            2*this_neg_image_windows * seed_image_windows';

  [table_neg_diff(neg_i,:), table_neg_idx(neg_i,:)] = min(diff,[],1); 
end
fprintf('Took %f seconds to process one image\n', toc(th));

% ------------------------------------------------------------------------
% evaluate score on top K
K = round(num_pos_images/2); score_list = zeros(num_seed_windows,1);
for seed_win_id = 1:num_seed_windows
  [~, idx] = sort( [table_pos_diff(:,seed_win_id); ...
                    table_neg_diff(:,seed_win_id)], 'ascend');
  incorrect  = sum(idx(1:K) > num_pos_images);
  score_list(seed_win_id) = (K-incorrect)/K; 
end

[~, ranked_idx] = sort(score_list, 'descend');

% save top Q% percent of all windows
top_Q = ceil(length(ranked_idx) * Q/100);

ranked_idx_top     = ranked_idx(1:top_Q);
score_top          = score_list(ranked_idx_top);
score_full         = score_list;

table_pos_idx_top  = table_pos_idx( :, ranked_idx_top);
table_pos_diff_top = table_pos_diff(:, ranked_idx_top);
table_neg_idx_top  = table_neg_idx( :, ranked_idx_top);
table_neg_diff_top = table_neg_diff(:, ranked_idx_top);
                                      
save(save_filename, 'ranked_idx_top', 'score_top', 'score_full',...
      'table_pos_idx_top', 'table_pos_diff_top', 'table_neg_idx_top', ...
      'table_neg_diff_top');

disp('done');
