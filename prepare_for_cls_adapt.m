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
for i = 1:length(classes)
    [imlist,flag(:,i)] = textread(sprintf(VOCopts.clsimgsetpath,classes{i},'trainval'),'%s %d');
end
fid = fopen('train_list.txt','w');
for i = 1:length(imlist)
    for j = 1:length(classes)
        if flag(i,j) ~= -1
            fprintf(fid, '%s %d\n', sprintf(VOCopts.imgpath,imlist{i}), 2*(j-1));
        else
            fprintf(fid, '%s %d\n', sprintf(VOCopts.imgpath,imlist{i}), 2*j-1);
        end
    end
end
fclose(fid);
