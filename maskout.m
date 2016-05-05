clear; close all; clc;

net_file = 'caffenet_cls_adapt_iter_10000.caffemodel';
crop_mode = 'warp';
crop_padding = 16;
VOCdevkit = './data/VOCdevkit2007';
imdb_train = imdb_from_voc(VOCdevkit, 'trainval', '2007');
cache_fc8_features(imdb_train, ...
    'crop_mode', crop_mode, ...
    'crop_padding', crop_padding, ...
    'net_file', net_file, ...
    'cache_name', 'voc_2007_trainval_fc8');

load('./imdb/cache/imdb_voc_2007_trainval.mat');
load('./imdb/cache/roidb_voc_2007_trainval.mat');
IND = cell(length(imdb.image_ids),1);
for i = 1:length(imdb.image_ids)
    d = load(['./cache/voc_2007_trainval_fc8/' imdb.image_ids{i} '.mat']);
    gtcls = unique(d.class(d.gt==1));
    feat_in = d.feat_in(d.gt~=1,1:2:end-1);
    feat_out = d.feat_out(d.gt~=1,1:2:end-1);
    bbs = d.boxes(d.gt~=1,:);
    dist = feat_in - feat_out;
    for cls = gtcls(:)'
        bbs_nms = nms([bbs,dist(:,cls)],0.3);
        num_top = min(size(bbs_nms,1),50);
        IND{i} = [IND{i};bbs_nms(1:num_top)];
    end
end
save('results_maskout_regions.mat','IND');
exit;
