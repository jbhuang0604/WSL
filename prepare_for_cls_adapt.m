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