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

function mil_region_mining(models, testset, year)

conf = voc_config();
cachedir = conf.paths.model_dir;                  
VOCopts  = conf.pascal.VOCopts;
load('class_pos_images.mat');
classid = strmatch(models{1}.class,VOCopts.classes,'exact');
image_ids = class_pos_images(classid).ids;
feat_opts = models{1}.opts;
ws = cat(2, cellfun(@(x) x.w, models, 'UniformOutput', false));
ws = cat(2, ws{:});
bs = cat(2, cellfun(@(x) x.b, models, 'UniformOutput', false));
bs = cat(2, bs{:});
boxes = cell(length(image_ids), 1);
for i = 1:length(image_ids)
    fprintf('%s: region mining: %d/%d\n', procid(), i, length(image_ids));
    d = load_cached_features_hos(testset, year, image_ids{i});
    d.feat = xform_feat_custom(d.feat, feat_opts);
    zs = bsxfun(@plus, d.feat*ws, bs);
    z = zs(d.gt~=1);
    [val, ind] = sort(z,'descend');
    bbs = d.boxes(d.gt~=1,:);
    boxes{i} = cat(2, single(bbs(ind(1),:)), z(ind(1)));
end
save_file = [cachedir models{1}.class '_best_boxes_' testset '_' year '.mat'];
save(save_file, 'boxes');
if ~exist('results_mil','file')
    mkdir('results_mil');
end
res_fn = ['./results_mil/' models{1}.class '_' testset '.txt'];
fid = fopen(res_fn, 'w');
for i = 1:length(image_ids)
    bbox = boxes{i};
    fprintf(fid, '%s %f %d %d %d %d\n', image_ids{i}, bbox(end), bbox(1:4));
end
fclose(fid);
