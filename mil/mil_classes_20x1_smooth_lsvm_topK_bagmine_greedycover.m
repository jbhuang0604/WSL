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

function mil_classes_20x1_smooth_lsvm_topK_bagmine_greedycover(classid)
                  
if ischar(classid),     classid = str2double(classid); end

conf = voc_config();

sharpness = '100';
loss_type = 'SmoothHinge';

svm_C = '0.001';
bias_mult = '10';
pos_loss_weight = '2';

class_list = conf.pascal.VOCopts.classes;

% learning parameters
svm_mu = 0.01;
topK = 15;
alpha = 0.95;
K1 = 0.5;
K2 = 1.0;
nms_threshold = 0.3;

load_filename = ['latentiter_*' ,...
  '_clss_' class_list{classid}, ... 
  '_C_'  svm_C, ...
  '_B_'  bias_mult, ...
  '_w1_' pos_loss_weight, ... 
  '_losstype_' loss_type, ...
  '_sharpness_' sharpness, ...
  '_mu_' num2str(svm_mu), ...
  '_alpha_' num2str(alpha),...
  '_K1_' num2str(K1), ...
  '_K2_' num2str(K2), ...
  '_nms_' num2str(nms_threshold), ...
  '_topK_' num2str(topK),...
  '_20x1_smooth_topK_bagmine_greedycover_final.mat'];

iterations = [];
d = dir([conf.paths.model_dir, load_filename]);
for i = 1:length(d)
  this_name = d(i).name;
  bars = strfind(this_name, '_');
  iteration_id = str2double(this_name(bars(1)+1 : bars(2)-1));
  iterations = [iterations; iteration_id];
end

highest_iteration_fileid = find(iterations == max(iterations));
load_filename = d(highest_iteration_fileid).name;

load([conf.paths.model_dir, load_filename], 'models');
mil_region_mining(models, 'trainval', '2007');
