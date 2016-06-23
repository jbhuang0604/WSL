% AUTORIGHTS
% ---------------------------------------------------------
% Copyright (c) 2016, Dong Li
% 
% This file is part of the WSL code and is available 
% under the terms of the MIT License provided in 
% LICENSE. Please retain this notice and LICENSE if you use 
% this file (or any portion of it) in your project.
% ---------------------------------------------------------

clear; close all; clc;
voc_init;
classes = VOCopts.classes;

% save image id for each class
try
    load('./mil/class_pos_images.mat');
catch
    for cls = 1:length(classes)
        [ids,flag] = textread(sprintf(VOCopts.clsimgsetpath,classes{cls},'trainval'),'%s %d');
        count = 0;
        for i = 1:length(ids)
            if flag(i) ~= -1
                count = count + 1;
                class_pos_images(cls).ids(count) = ids(i);
            end
        end
    end
    save('./mil/class_pos_images.mat','class_pos_images');
end

% extract fc7 features
net_file = 'caffenet_cls_adapt_iter_10000.caffemodel';
crop_mode = 'warp';
crop_padding = 16;
VOCdevkit = './data/VOCdevkit2007';
imdb_train = imdb_from_voc(VOCdevkit, 'trainval', '2007');
cache_fc7_features(imdb_train, ...
    'crop_mode', crop_mode, ...
    'crop_padding', crop_padding, ...
    'net_file', net_file, ...
    'cache_name', 'voc_2007_trainval_fc7');

% empoly mil to mine confident regions
tmp = pwd;
cd('mil');
for cls = 1:length(class_pos_images)
    num = length(class_pos_images(cls).ids);
    for i = 1:num
        cluster_patches_parallel_single_nogt_20x1(i,cls);
    end
    models = train_classes_20x1_smooth_greedycover(cls);
    models = train_classes_20x1_smooth_lsvm_topK_bagmine_greedycover(cls);
    mil_classes_20x1_smooth_lsvm_topK_bagmine_greedycover(cls);
end
cd(tmp);
