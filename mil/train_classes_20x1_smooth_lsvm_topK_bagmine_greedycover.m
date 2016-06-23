% AUTORIGHTS
% ---------------------------------------------------------
% Copyright (c) 2014, Hyun Oh Song
% Copyright (c) 2016, Dong Li
% 
% This file is part of the WSL code and is available 
% under the terms of the MIT License provided in 
% LICENSE. Please retain this notice and LICENSE if you use 
% this file (or any portion of it) in your project.
% ---------------------------------------------------------

function models = train_classes_20x1_smooth_lsvm_topK_bagmine_greedycover(...
  classid, varargin)
     
% represents both positive and negative images as bags and refines initial detector by minimizing smooth latent svm loss.

% initialize and fix the random seed
randn('state', 1);
rand('state',  1);

% addpath
addpath(genpath('minFunc_2012/'));
addpath('projsplx/');

trainset = 'trainval';
year     = '2007';

if ischar(classid),   classid   = str2double(classid);   end

% cast optional parameters into double
if length(varargin) ~= 0
  for i = 1:length(varargin)
    if mod(i,2)==0
      if ischar(varargin{i}), varargin{i} = str2double(varargin{i}); end
    end
  end
end     

ip = inputParser;
ip.addRequired('trainset', @isstr);
ip.addRequired('year',     @isstr);
ip.addRequired('classid',  @isscalar);

ip.addParamValue('sharpness', 100, @isscalar);
ip.addParamValue('svm_mu', 0.01, @isscalar);
ip.addParamValue('topK', 15, @isscalar);
ip.addParamValue('alpha', 0.95, @isscalar);
ip.addParamValue('K1', 0.5, @isscalar);
ip.addParamValue('K2', 1.0, @isscalar);
ip.addParamValue('nms_threshold', 0.3, @isscalar);

ip.addParamValue('loss_type',    'SmoothHinge', @isstr);
ip.addParamValue('svm_C',           10^-3,   @isscalar);
ip.addParamValue('bias_mult',       10,      @isscalar);
ip.addParamValue('pos_loss_weight', 2,       @isscalar);
ip.addParamValue('layer',           'fc7', @isstr);
ip.addParamValue('fine_tuned',      0,       @isscalar);
ip.addParamValue('use_flipped',     0,       @isscalar);                
ip.addParamValue('target_norm', 20, @isscalar);

ip.parse(trainset, year, classid, varargin{:});
opts = ip.Results;

fprintf('\n\n~~~~~~~~~~~~~~~~~~~~~~~~~~~~\n');
fprintf('Training options:\n');
disp(opts);
fprintf('~~~~~~~~~~~~~~~~~~~~~~~~~~~~\n\n'); 

conf = voc_config();
clss = conf.pascal.VOCopts.classes;
clss = clss(classid);
num_clss = 1;

dataset.year = year;
dataset.trainset = trainset;
dataset.image_ids = textread(sprintf(conf.pascal.VOCopts.imgsetpath, trainset), '%s');
dataset.image_ids_small = dataset.image_ids(randperm(length(dataset.image_ids), 1000));
dataset.pos_image_ids = dataset.image_ids;

% ------------------------------------------------------------------------
load('class_pos_images.mat');
pos_image_ids = class_pos_images(classid).ids;

neg_image_ids = setdiff(dataset.image_ids, pos_image_ids);

dataset.neg_image_ids = neg_image_ids;

% sample_pos_image_ids = pos_image_ids(1:40);
% sample_neg_image_ids = neg_image_ids(1:200);
sample_pos_image_ids = pos_image_ids;
sample_neg_image_ids = neg_image_ids;

num_pos_images = length(sample_pos_image_ids);
num_neg_images = length(sample_neg_image_ids);
% ------------------------------------------------------------------------

% ------------------------------------------------------------------------
% Get or compute the average norm of the features
if ~exist('feat_stats','file')
    mkdir('feat_stats');
end
save_file = sprintf('feat_stats/stats_%s_%s_layer_%s_finetuned_%d', ...
                    trainset, year, opts.layer, opts.fine_tuned);
try
  ld = load(save_file);
  opts.feat_norm_mean = ld.feat_norm_mean;
  clear ld;
catch
  [feat_norm_mean, stddev] = feat_stats_hos(trainset, year, opts.layer, opts.fine_tuned);
  save(save_file, 'feat_norm_mean', 'stddev');
  opts.feat_norm_mean = feat_norm_mean;
end
fprintf('average norm = %.3f\n', opts.feat_norm_mean);
% ------------------------------------------------------------------------

% ------------------------------------------------------------------------
% Load initial classifier
latent_iter = 0;

load([conf.paths.model_dir, ...
  'latentiter_' num2str(latent_iter) '_clss_' clss{1}, ... 
  '_C_'  num2str(opts.svm_C), ...
  '_B_'  num2str(opts.bias_mult), ...
  '_w1_' num2str(opts.pos_loss_weight), ... 
  '_losstype_' opts.loss_type,...
  '_sharpness_' num2str(opts.sharpness),...
  '_alpha_' num2str(opts.alpha),...
  '_K1_' num2str(opts.K1), ...
  '_K2_' num2str(opts.K2), ...
  '_nms_' num2str(opts.nms_threshold), ...
  '_20x1_smooth_greedycover_final.mat'], 'models'); 
% ------------------------------------------------------------------------

% ------------------------------------------------------------------------
% Train with hard negative mining
max_latent_iter = 10;

th = tic(); % measure training time

for latent_iter = 1:max_latent_iter
  fprintf('latent positive update %d\n', latent_iter);  
  
  save_filename = [conf.paths.model_dir, ...
    'latentiter_' num2str(latent_iter) '_clss_' clss{1}, ... 
    '_C_'  num2str(opts.svm_C), ...
    '_B_'  num2str(opts.bias_mult), ...
    '_w1_' num2str(opts.pos_loss_weight), ... 
    '_losstype_' opts.loss_type, ...
    '_sharpness_' num2str(opts.sharpness), ...
    '_mu_' num2str(opts.svm_mu), ...
    '_alpha_' num2str(opts.alpha),...
    '_K1_' num2str(opts.K1),...
    '_K2_' num2str(opts.K2),...
    '_nms_' num2str(opts.nms_threshold),....
    '_topK_' num2str(opts.topK), ...
    '_20x1_smooth_topK_bagmine_greedycover_final.mat'];
  
  % check if we already computed the file.
  if exist(save_filename, 'file') ~= 0
    load(save_filename, 'models');
    fprintf('models exists for latent iter %d loaded.\n', latent_iter);
  else
    % Get top K windows in all positive examples into the cache  
    X_pos = get_all_features_topK_bagmine(...
                  models, dataset, opts, sample_pos_image_ids);

    % Get top K windows in all negative examples into the cache
    X_neg = get_all_features_topK_bagmine(...
                  models, dataset, opts, sample_neg_image_ids);

    % Update model       
    model_before = [models{1}.w; models{1}.b];
    models{1} = update_model_smooth_latent_light(models{1}, X_pos, X_neg, ...
                                        num_pos_images, num_neg_images, opts);
    model_after = [models{1}.w; models{1}.b];
   
    % check if model converged 
    if norm(model_after - model_before) < 1e-8
      fprintf('Latent positive relabeling convergence detected. Breaking.\n');
      break;
    end

    save(save_filename, 'models'); 
  end
end

fprintf('Took %.3f hours to train\n', toc(th)/3600);        
  

% ------------------------------------------------------------------------
function X = get_all_features_topK_bagmine(...
                             models, dataset, opts, sample_image_ids)
% ------------------------------------------------------------------------

d = load_cached_features_hos(1, dataset.trainset, dataset.year, sample_image_ids{1});

feat_dim = size(d.feat,2);      
X = zeros(feat_dim, opts.topK * length(sample_image_ids), 'single');
  
start_i = 1;
for i = 1:length(sample_image_ids)
  d = load_cached_features_hos(1, dataset.trainset, dataset.year, sample_image_ids{i});
        
  d.feat = xform_feat_custom(d.feat, opts);   
  
  % remove all ground truth boxes
  d.feat  = d.feat( d.gt ~= 1, :);
  
  if size(d.feat,1) < opts.topK
      continue;
  end

  zs = d.feat * models{1}.w + models{1}.b;
  
  [~, top_ids] = sort(zs, 'descend');
  sel = top_ids(1 : opts.topK);
  
  end_i = start_i + opts.topK - 1;
  X(:, start_i : end_i) = d.feat(sel,:)';
  start_i = end_i + 1;
end
disp('done');
        
% ------------------------------------------------------------------------
function model = update_model_smooth_latent_light(model, X_pos, X_neg, ...
                              num_pos_bags, num_neg_bags, opts)
% ------------------------------------------------------------------------
    
pweight = opts.pos_loss_weight;

[pos_cum_bag_idx, pos_averaging_matrix] = prebuild_averaging_matrix(...
                            opts.topK * ones(num_pos_bags, 1), num_pos_bags);
[neg_cum_bag_idx, neg_averaging_matrix] = prebuild_averaging_matrix(...
                            opts.topK * ones(num_neg_bags, 1), num_neg_bags);
                                         
 % prebuild label, and pweighted labels
y = [ones(num_pos_bags,1); -ones(num_neg_bags,1)];
pweighted_y = [pweight*y(1:num_pos_bags); y(num_pos_bags+1:end)];

options.Method  = 'lbfgs';
options.Display = 'OFF'; %no output, default = 2;

% w0 depends on whether this is a latent run or not
if isempty(model.w)
    error('In latent runs, model should never be empty');
else
    w0 = double([model.w; model.b/opts.bias_mult]);
    [cost,~] = slslvm_cost_smoothhinge_bagmine(...
      w0, X_pos, X_neg, pos_averaging_matrix, pos_cum_bag_idx, ...
      neg_averaging_matrix, neg_cum_bag_idx,...
      num_pos_bags, num_neg_bags, y, pweighted_y, opts.svm_C, opts.svm_mu, ...
      pweight, opts.sharpness, opts.bias_mult);
    fprintf('cost before lbfgs: %.4f\n', cost);
end

if strcmp(opts.loss_type, 'L1hinge')
  error('dense version not implemented yet');

elseif strcmp(opts.loss_type, 'SmoothHinge')
    w_opt = minFunc(@(w) slslvm_cost_smoothhinge_bagmine(...
                w, X_pos, X_neg, pos_averaging_matrix, pos_cum_bag_idx, ...
                neg_averaging_matrix, neg_cum_bag_idx,...
                num_pos_bags, num_neg_bags, y, pweighted_y, opts.svm_C, opts.svm_mu, ...
                pweight, opts.sharpness, opts.bias_mult), w0, options);
              
    [cost,~] = slslvm_cost_smoothhinge_bagmine(...
      w_opt, X_pos, X_neg, pos_averaging_matrix, pos_cum_bag_idx, ...
      neg_averaging_matrix, neg_cum_bag_idx,...
      num_pos_bags, num_neg_bags, y, pweighted_y, opts.svm_C, opts.svm_mu, ...
      pweight, opts.sharpness, opts.bias_mult);
    fprintf('cost after lbfgs: %.4f\n', cost);
                  
else
    error('Unrecognized loss');
end

model.w = single(w_opt(1:end-1));
model.b = single(w_opt(end)*opts.bias_mult);


% ------------------------------------------------------------------------
function [cum_bag_idx, averaging_matrix] = prebuild_averaging_matrix(...
                                    num_insts_per_bag, bag_size)
% ------------------------------------------------------------------------

cum_bag_idx = cumsum(num_insts_per_bag);
if size(cum_bag_idx,1) ~= 1
  cum_bag_idx = cum_bag_idx';
end

% precompute averaging matrix
averaging_matrix = zeros(cum_bag_idx(end), bag_size);
i_start = 1;
for i = 1:bag_size
    i_end = cum_bag_idx(i);
    averaging_matrix( i_start : i_end, i) = ...
        ones(num_insts_per_bag(i),1);
    i_start = i_end + 1;
end

