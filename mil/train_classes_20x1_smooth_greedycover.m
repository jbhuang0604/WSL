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

function models = train_classes_20x1_smooth_greedycover(classid, varargin)
                            
% HOS: Sort clusters by discriminativeness score, greedily take
%     non-overlapping (image ids, or boxes) in K1 until the score goes bad          

% HOS: initialize and fix the random seed
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
ip.addParamValue('alpha', 0.95, @isscalar);
ip.addParamValue('K1', 0.5, @isscalar);
ip.addParamValue('K2', 1.0, @isscalar);
ip.addParamValue('nms_threshold', 0.3, @isscalar);

ip.addParamValue('loss_type', 'SmoothHinge', @isstr);
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
% Init models
models = {};
for i = 1:num_clss
  models{i} = init_model(clss{i}, dataset, conf, opts);
end
% ------------------------------------------------------------------------

% ------------------------------------------------------------------------
% Get all positive examples
X_pos = get_positive_features_paris_greedycover( ...
                    models, dataset, opts, classid, sample_pos_image_ids );
                  
for i = 1:num_clss
  fprintf('%14s has %6d positive instances\n', models{i}.class, size(X_pos{i},1));
  X_pos{i}    = xform_feat_custom(X_pos{i}, opts);
end
% ------------------------------------------------------------------------

% ------------------------------------------------------------------------
% Init training caches
caches = {};
for i = 1:num_clss
  caches{i} = init_cache(models{i}, X_pos{i});
end
% ------------------------------------------------------------------------

% ------------------------------------------------------------------------
% Train with hard negative mining
first_time = true;
force_update = false;
max_hard_epochs = 1;
max_latent_iter = 0;       

th = tic(); % measure training time

for latent_iter = 0:max_latent_iter
  % Latent positive relabling
  if latent_iter > 0
    fprintf('latent positive update %d\n', latent_iter);
    %X_pos = get_positive_features_paris_maxcover(models, dataset, true, opts,...
    %                 classid, sample_pos_image_ids, sample_neg_image_ids);
    for i = 1:num_clss
      caches{i}.X_pos    = X_pos{i};
      fprintf('%14s has %10d positive instances\n', ...
              models{i}.class, size(X_pos{i},1));
    end
    % force model update
    force_update = true;
  end
  
  % Train lbfgs SVMs with hard negative mining
  for hard_epoch = 1:max_hard_epochs
    for i = 1:length(dataset.neg_image_ids)
      fprintf('%s: hard neg epoch %d %d/%d\n', ...
              procid(), hard_epoch, i, length(dataset.neg_image_ids));

      % Get hard negatives for all classes at once (avoids loading feature cache
      % more than once)
      [X, keys] = sample_negative_features(first_time, models, caches, dataset, i, opts);

      % Add sampled negatives to each classes training cache, removing
      % duplicates
      for j = 1:num_clss
        if ~isempty(keys{j})
          if isempty(caches{j}.keys)
              dups = [];
          else
              [~, ~, dups] = intersect(caches{j}.keys, keys{j}, 'rows');
          end
          assert(isempty(dups));
          caches{j}.X_neg = cat(1, caches{j}.X_neg, X{j});
          caches{j}.keys  = cat(1, caches{j}.keys, keys{j});
          caches{j}.num_added = caches{j}.num_added + size(keys{j},1);
        end
        
        % Update model if
        %  - first time seeing negatives
        %  - more than retrain_limit negatives have been added
        %  - its the final image of the final epoch
        is_last_time = (hard_epoch == max_hard_epochs && i == length(dataset.neg_image_ids));
        hit_retrain_limit = (caches{j}.num_added > caches{j}.retrain_limit);
        if force_update || first_time || hit_retrain_limit || is_last_time
          fprintf('  Retraining %s model\n', models{j}.class);
          fprintf('    Cache holds %d pos examples %d neg examples\n', ...
                  size(caches{j}.X_pos,1), size(caches{j}.X_neg,1));
                
          models{j} = update_model(models{j}, caches{j}, opts);
          
          caches{j}.num_added = 0;

          z_pos = caches{j}.X_pos*models{j}.w + models{j}.b;
          z_neg = caches{j}.X_neg*models{j}.w + models{j}.b;

          caches{j}.pos_loss(end+1) = opts.svm_C*sum(max(0, 1 - z_pos))*opts.pos_loss_weight;
          caches{j}.neg_loss(end+1) = opts.svm_C*sum(max(0, 1 + z_neg));
          caches{j}.reg_loss(end+1) = 0.5*models{j}.w'*models{j}.w + ...
                                      0.5*(models{j}.b/opts.bias_mult)^2;
          caches{j}.tot_loss(end+1) = caches{j}.pos_loss(end) + ...
                                      caches{j}.neg_loss(end) + ...
                                      caches{j}.reg_loss(end);

          for t = 1:length(caches{j}.tot_loss)
            fprintf('    %2d: obj val: %.3f = %.3f (pos) + %.3f (neg) + %.3f (reg)\n', ...
                    t, caches{j}.tot_loss(t), caches{j}.pos_loss(t), ...
                    caches{j}.neg_loss(t), caches{j}.reg_loss(t));
          end

          % evict easy examples
          easy = find(z_neg < caches{j}.evict_thresh);
          caches{j}.X_neg(easy,:) = [];
          caches{j}.keys(easy,:) = [];
          fprintf('    Pruning easy negatives\n');
          fprintf('    Cache holds %d pos examples %d neg examples\n', ...
                  size(caches{j}.X_pos,1), size(caches{j}.X_neg,1));
          fprintf('    %d pos support vectors\n', numel(find(z_pos <=  1)));
          fprintf('    %d neg support vectors\n', numel(find(z_neg >= -1)));

          %model = models{j};
          %save([conf.paths.model_dir models{j}.class '_' num2str(length(caches{j}.tot_loss))], 'model');
          %clear model;
        end
      end
      first_time = false;
      force_update = false;
    end
  end
end

save([conf.paths.model_dir, ...
  'latentiter_' num2str(latent_iter) '_clss_' clss{1}, ... 
  '_C_'  num2str(opts.svm_C), ...
  '_B_'  num2str(opts.bias_mult), ...
  '_w1_' num2str(opts.pos_loss_weight), ... 
  '_losstype_' opts.loss_type, ...
  '_sharpness_' num2str(opts.sharpness), ...
  '_alpha_' num2str(opts.alpha), ...
  '_K1_' num2str(opts.K1), ...
  '_K2_' num2str(opts.K2), ...
  '_nms_' num2str(opts.nms_threshold), ...
  '_20x1_smooth_greedycover_final.mat'], 'models'); 

fprintf('Took %.3f hours to train\n', toc(th)/3600);  


% ------------------------------------------------------------------------
function [X_neg, keys] = sample_negative_features(first_time, models, ...
                                                  caches, dataset, ind, ...
                                                  opts)
% ------------------------------------------------------------------------
d = load_cached_features_hos(dataset.trainset, dataset.year, dataset.neg_image_ids{ind});

if length(d.overlap) ~= size(d.feat, 1)
  fprintf('WARNING: %s has data mismatch\n', dataset.neg_image_ids{ind});
  X_neg = cell(1, length(models));
  keys = cell(1, length(models));
  return;
end

d.feat = xform_feat_custom(d.feat, opts);
%d.feat = o2p(d.feat);

neg_ovr_thresh = 0.3;

if first_time
  for i = 1:length(models)
    %I = find(d.overlap(:, models{i}.class_id) < neg_ovr_thresh);
    I = (1:size(d.feat,1))';
    X_neg{i} = d.feat(I,:);
    keys{i}  = [ind*ones(length(I),1) I];
  end
else
  ws = cat(2, cellfun(@(x) x.w, models, 'UniformOutput', false));
  ws = cat(2, ws{:});
  bs = cat(2, cellfun(@(x) x.b, models, 'UniformOutput', false));
  bs = cat(2, bs{:});
  zs = bsxfun(@plus, d.feat*ws, bs);
  for i = 1:length(models)
    z = zs(:,i);
%     I = find((z > caches{i}.hard_thresh) & ...
%              (d.overlap(:, models{i}.class_id) < neg_ovr_thresh));

    I = find(z > caches{i}.hard_thresh);

    % apply NMS to scored boxes
    % select as negatives anything that survived NMS
    % and has < 50% overlap with postives of this class
    % and is violating the margin

%    boxes = cat(2, single(d.boxes), z);
%    nms_keep = false(size(boxes,1), 1);
%    nms_keep(nms(boxes, 0.3)) = true;
%
%    I = find((z > caches{i}.hard_thresh) & ...
%             (nms_keep == true) & ...
%             (d.overlap(:, models{i}.class_id) < neg_ovr_thresh));

    % Avoid adding duplicate features
    keys_ = [ind*ones(length(I),1) I];
    if isempty(caches{i}.keys) || isempty(keys_)
        dups = [];
    else
        [~, ~, dups] = intersect(caches{i}.keys, keys_, 'rows');
    end
    keep = setdiff(1:size(keys_,1), dups);
    I = I(keep);

    % Unique hard negatives
    X_neg{i} = d.feat(I,:);
    keys{i} = [ind*ones(length(I),1) I];
  end
end


% ------------------------------------------------------------------------
function model = update_model(model, cache, opts)
% ------------------------------------------------------------------------
num_pos  = size(cache.X_pos, 1);
num_neg  = size(cache.X_neg, 1);
feat_dim = size(cache.X_pos, 2);

pweight = opts.pos_loss_weight;
X = zeros( num_pos*pweight+num_neg, feat_dim+1);

X(1:num_pos*pweight, 1:end-1) = repmat(cache.X_pos,pweight,1);
X(num_pos*pweight+1:end, 1:end-1) = cache.X_neg;
% augment the bias feature * opt.bias_mult factor
X(:, end) = opts.bias_mult * ones(1, num_pos*pweight+num_neg);
y = cat(1, repmat(ones(num_pos,1),pweight,1), -ones(num_neg,1));

options.Method  = 'lbfgs';
options.Display = 'OFF'; %no output, default = 2;

if isempty(model.w)
    w0 = zeros(feat_dim+1,1); 
else
    w0 = double([model.w; model.b/opts.bias_mult]);
end

if strcmp(opts.loss_type, 'L1hinge')
    w_opt = minFunc(@(w) SVM_Cost_L1hinge(...
                w, X, y, opts.svm_C), w0, options);

elseif strcmp(opts.loss_type, 'SmoothHinge')
    w_opt = minFunc(@(w) SVM_Cost_SmoothHinge(...
                w, X, y, opts.svm_C, opts.sharpness), w0, options);
            
elseif strcmp(opts.loss_type, 'Logistic')
    w_opt = minFunc(@(w) SVM_Cost_Logistic(...
                w, X, y, opts.svm_C), w0, options);
end

model.w = single(w_opt(1:end-1));
model.b = single(w_opt(end)*opts.bias_mult);


% ------------------------------------------------------------------------
function X_pos = get_positive_features_paris_greedycover(...
                     models, dataset, opts, classid, sample_pos_image_ids)
% ------------------------------------------------------------------------
% HOS: Use maxcover to create a pool of positive windows.
%      \alpha controls max_coverage 
%      \K1 controls number of nearest neighbors per cluster
%      \K2 controls number of positive boxes to take per cluster

% A. Construct positive training set from max cover

% trainset_matrix is a 2 by #boxes matrix (pos image ids; box ids)
trainset_matrix = [];
num_pos_images = length(sample_pos_image_ids);

% construct graph matrix, # clusters by # pos images 
[graph_image_matrix, graph_box_matrix, ...
    per_cluster_paris_score_K1, per_cluster_paris_score_K2] = ...
                  construct_graph(classid, sample_pos_image_ids, opts);
                
coverage = 0; popped_cluster_history = [];  
per_cluster_paris_score_copy = per_cluster_paris_score_K1; %copy for plotting
% go down sorted list of paris scores and keep taking unclaimed positive
% images in top K1 (but take boxes in top K2)
while coverage < (opts.alpha * num_pos_images)
  
  % pop the cluster with best paris score
  c_top = find(per_cluster_paris_score_K1 == max(per_cluster_paris_score_K1));
  
  % take care of ties here
  if length(c_top) > 1
    fprintf('warning: tie detected!\n');
    % compute paris score @ K2 to break the tie
    
    c_top_scores = per_cluster_paris_score_K2(c_top);
    c_top = c_top( find(c_top_scores == max(c_top_scores)) ); 
    
    % still tied? just take the first one.
    c_top = c_top(1);
  end
  
  % push boxes for cluster's activated pos iamges to train set, 
  box_ids = graph_box_matrix(c_top, :);
  trainset_matrix = [trainset_matrix, [find(box_ids~=0); box_ids(box_ids~=0)] ];
  
  % update graph: remove activated pos images from c_top
  activated_images = find(graph_image_matrix(c_top,:) ~= false);
  graph_image_matrix(:, activated_images) = false;
  graph_box_matrix(:, activated_images) = 0;
  
  % void this cluster from popping up again
  per_cluster_paris_score_K1(c_top) = -inf;
  popped_cluster_history = [popped_cluster_history, c_top];  
  
  % update coverage
  coverage = coverage + length(activated_images);
  
  fprintf('[%d] current coverage: %d, cluster %d covered %d\n', ...
    length(popped_cluster_history), coverage, c_top, length(activated_images));
end

% B. Decode trainset_matrix and create positive feature matrix
%      make sure to preserve original order of trainset_matrix

% remove exact duplicates in trainset_matrix
trainset_matrix = remove_exact_duplicate_columns_preserve_order(trainset_matrix);

X_pos = cell(length(models),1);
X_pos{1} = single([]);

% remove highly overlapping boxes with NMS 0.3
num_suppressed_boxes = 0;
assigned_boxes(num_pos_images).coords = [];
for boxid = 1:length(trainset_matrix)
  % load this pos image's features and boxes; check if indexing is correct
  this_image_idx = trainset_matrix(1, boxid);
  this_box_idx   = trainset_matrix(2, boxid);
  
  img_struct = load_cached_features_hos(dataset.trainset, dataset.year, sample_pos_image_ids{this_image_idx});
  
  img_struct.boxes = img_struct.boxes(img_struct.gt~=1,:);
  img_struct.feat  = img_struct.feat( img_struct.gt~=1,:); 
  
  this_feature = img_struct.feat( this_box_idx, :);
  this_box     = img_struct.boxes(this_box_idx, :);
  
  % check if there's box already taken and nms if exists.
  is_suppressed = check_nms_with_existing_boxes(...
                    assigned_boxes(this_image_idx).coords, this_box, opts);
  if is_suppressed
    num_suppressed_boxes = num_suppressed_boxes + 1;
    continue;
  end
  
  assigned_boxes(this_image_idx).coords = [...
                     assigned_boxes(this_image_idx).coords; this_box];
                   
  X_pos{1} = cat(1, X_pos{1}, this_feature);
end
fprintf(['done creating positive feature matrix.\n',...
      '# pos images: %d, # passed boxes: %d, # suppresed boxes: %d\n'], ...
      num_pos_images, size(X_pos{1},1), num_suppressed_boxes);


% ------------------------------------------------------------------------
function matrix = remove_exact_duplicate_columns_preserve_order(matrix)
% ------------------------------------------------------------------------
num_boxes_before = size(matrix,2);

[newmat, newids] = unique(matrix', 'rows', 'first');
hasDuplicates = size(newmat,1) < num_boxes_before;
if hasDuplicates
  dupColumns = setdiff(1:num_boxes_before, newids);
  matrix(:,dupColumns) = [];
end

num_boxes_after = size(matrix,2);
fprintf('removed %d exact duplicates\n', num_boxes_before-num_boxes_after);


% ------------------------------------------------------------------------
function [graph_image_matrix, graph_box_matrix, ...
            per_cluster_paris_score_K1, per_cluster_paris_score_K2] = ...
                       construct_graph(classid, sample_pos_image_ids, opts)
% ------------------------------------------------------------------------
%      \K1 controls number of nearest neighbors per cluster
%      \K2 controls number of positive boxes to take per cluster

num_pos_images = length(sample_pos_image_ids);

% count number of saved clusters
num_clusters = 0;
for seed_pos_image_id = 1:num_pos_images
  save_filename = sprintf('paris_results_nogt_20x1/%s_%d.mat',...
                       sample_pos_image_ids{seed_pos_image_id}, classid);
  
  load(save_filename, 'score_top');
  num_clusters = num_clusters + length(score_top);
end

% form a binary graph matrix size: # clusters by # pos images 
% graph_image_matrix holds activated positive images in top K1
graph_image_matrix = false(num_clusters, num_pos_images);
% graph_box_matrix holds activated positive boxes in top K2
graph_box_matrix = zeros(num_clusters, num_pos_images, 'single');

% record per_cluster_paris score
per_cluster_paris_score_K1 = zeros(num_clusters, 1, 'single');
per_cluster_paris_score_K2 = zeros(num_clusters, 1, 'single');

% Loop over each positive images and grab pos image and box list
cid = 0;
for seed_pos_image_id = 1:num_pos_images
  save_filename = sprintf('paris_results_nogt_20x1/%s_%d.mat',...
                       sample_pos_image_ids{seed_pos_image_id}, classid);
  
  load(save_filename);  
  
  % loop through each boxes (= clusters)
  for seed_win_id = 1:length(score_top)
    cid = cid + 1;

    [~, image_idx] = sort( [table_pos_diff_top(:,seed_win_id);...
                            table_neg_diff_top(:,seed_win_id)], 'ascend');
                          
    % fill image matrix: restrict list pos images to top K1 NN
    image_idx_top_K1 = image_idx(1 : round(opts.K1*num_pos_images) );      
    % filter only positive images
    pos_image_list_K1 = image_idx_top_K1(image_idx_top_K1 <= num_pos_images);
    graph_image_matrix(cid, pos_image_list_K1) = true;
    % record paris score @ K1
    per_cluster_paris_score_K1(cid) = length(pos_image_list_K1) / ...
                                            round(opts.K1*num_pos_images);
    
                                          
    % fill box matrix: restrict list pos images to top K2 NN
    image_idx_top_K2 = image_idx(1 : round(opts.K2*num_pos_images) );    
    % filter only positive images
    pos_image_list_K2 = image_idx_top_K2(image_idx_top_K2 <= num_pos_images);    
    % record paris score @ K2
    per_cluster_paris_score_K2(cid) = length(pos_image_list_K2) / ...
                                            round(opts.K2*num_pos_images);
        
    % retreive box ids for each positives. 
    % convert pos image id -> box id in the given pos image
    pos_box_list_K2 = table_pos_idx_top(pos_image_list_K2, seed_win_id);
    graph_box_matrix(cid, pos_image_list_K2) = pos_box_list_K2;
  end
end
    

% ------------------------------------------------------------------------
function is_suppressed = check_nms_with_existing_boxes(...
                                          existing_boxes, this_box, opts)
% ------------------------------------------------------------------------

if isempty(existing_boxes)
  is_suppressed = false;
  return;
end

%nms_overlap_threshold = 0.3; % kill boxes if nms overlap > 0.3

assert( size(existing_boxes,2) == size(this_box,2));

num_existing_boxes = size(existing_boxes,1);

% parse existing boxes
x1 = existing_boxes(:,1);
y1 = existing_boxes(:,2);
x2 = existing_boxes(:,3);
y2 = existing_boxes(:,4);
area = (x2-x1+1) .* (y2-y1+1);

% parse this new box
tx1 = this_box(:,1);
ty1 = this_box(:,2);
tx2 = this_box(:,3);
ty2 = this_box(:,4);
new_box_area = (tx2-tx1+1) * (ty2 - ty1+1);

is_suppressed = false;

for j = 1:num_existing_boxes
  xx1 = max(x1(j), tx1);
  yy1 = max(y1(j), ty1);
  xx2 = min(x2(j), tx2);
  yy2 = min(y2(j), ty2);
  w = xx2-xx1+1;
  h = yy2-yy1+1;
    
  if w > 0 && h > 0
    % compute overlap
    inter = w*h;
    o = inter / (area(j) + new_box_area - inter);
    if o > opts.nms_threshold
      is_suppressed = true;
      break;
    end
  end
end

 
% ------------------------------------------------------------------------
function model = init_model(cls, dataset, conf, opts)
% ------------------------------------------------------------------------
model.class = cls;
%model.class_id = strmatch(model.class, conf.pascal.VOCopts.classes);
model.trainset = dataset.trainset;
model.year = dataset.year;
model.w = [];
model.b = [];
model.thresh = -1.1;
model.opts = opts;


% ------------------------------------------------------------------------
function cache = init_cache(model, X_pos)
% ------------------------------------------------------------------------
cache.X_pos    = X_pos;
cache.X_neg = single([]);
cache.keys = [];
cache.num_added = 0;
cache.retrain_limit = 2000;
cache.evict_thresh = -1.2;
cache.hard_thresh = -1.0001;
cache.pos_loss = [];
cache.neg_loss = [];
cache.reg_loss = [];
cache.tot_loss = [];
