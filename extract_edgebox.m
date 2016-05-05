clear; close all; clc;
voc_init;
classes = VOCopts.classes;

% load pre-trained edge detection model and set opts (see edgesDemo.m)
model=load('./edgebox/models/forest/modelBsds'); model=model.model;
model.opts.multiscale=0; model.opts.sharpen=2; model.opts.nThreads=4;
% set up opts for edgeBoxes (see edgeBoxes.m)
opts = edgeBoxes;
opts.alpha = .65;     % step size of sliding window search
opts.beta  = .75;     % nms threshold for object proposals
opts.minScore = .01;  % min score of boxes to detect
opts.maxBoxes = 2000;  % max number of boxes to detect

images = textread(sprintf(VOCopts.imgsetpath,'trainval'),'%s');
boxes = {};
for i = 1:length(images)
    fprintf('extract edgebox: voc2007 %s: %d/%d\n', 'trainval', i, length(images));
    im = imread(sprintf(VOCopts.imgpath,images{i}));
    if size(im,3)~=3
        im = cat(3,im,im,im);
    end
    bbs = edgeBoxes(im,model,opts);
    bbs_new = double([bbs(:,2),bbs(:,1),bbs(:,2)+bbs(:,4),bbs(:,1)+bbs(:,3)]);
    boxes{i} = bbs_new;
end
save('./data/edgebox_data/voc_2007_trainval.mat','boxes','images','-v7.3');

images = textread(sprintf(VOCopts.imgsetpath,'test'),'%s');
boxes = {};
for i = 1:length(images)
    fprintf('extract edgebox: voc2007 %s: %d/%d\n', 'test', i, length(images));
    im = imread(sprintf(VOCopts.imgpath,images{i}));
    if size(im,3)~=3
        im = cat(3,im,im,im);
    end
    bbs = edgeBoxes(im,model,opts);
    bbs_new = double([bbs(:,2),bbs(:,1),bbs(:,2)+bbs(:,4),bbs(:,1)+bbs(:,3)]);
    boxes{i} = bbs_new;
end
save('./data/edgebox_data/voc_2007_test.mat','boxes','images','-v7.3');
