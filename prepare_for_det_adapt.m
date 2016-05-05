clear; close all; clc;
voc_init;
classes = VOCopts.classes;
fid = fopen('all_dets.txt','w');
for cls = 1:length(classes)
    [ids,scores,x1,y1,x2,y2] = textread(['./mil/results_mil/' classes{cls} '_trainval.txt'],'%s %f %f %f %f %f');
    for i = 1:length(ids)
        fprintf(fid, '%s %f %f %f %f %f %d\n', ids{i},scores(i),x1(i),y1(i),x2(i),y2(i),cls);
    end
end
fclose(fid);

load('./imdb/cache/imdb_voc_2007_trainval.mat');
load('./imdb/cache/roidb_voc_2007_trainval.mat');
[ids,scores,x1,y1,x2,y2,cls]=textread('all_dets.txt','%s %f %f %f %f %f %d');
if ~exist('det_anno','file')
    mkdir('det_anno');
end
for i = 1:length(imdb.image_ids)
    ind = strmatch(imdb.image_ids{i},ids);
    num = length(ind);
    fid = fopen(['./det_anno/' imdb.image_ids{i} '.txt'],'w');
    if num ~= 0 
        fprintf(fid,'%d\n',num);
        for j = 1:num
            fprintf(fid,'%f %f %f %f %d\n',x1(ind(j)),y1(ind(j)),x2(ind(j)),y2(ind(j)),cls(ind(j)));
        end
    end
    fclose(fid);
end

load('./imdb/cache/imdb_voc_2007_test.mat');
load('./imdb/cache/roidb_voc_2007_test.mat');
for i = 1:length(imdb.image_ids)
    r = roidb.rois(i);
    ind = find(r.gt~=0);
    bbs = r.boxes(ind,:);
    num = length(ind);
    fid = fopen(['./det_anno/' imdb.image_ids{i} '.txt'],'w');
    fprintf(fid,'%d\n',num);
    for j = 1:num
        fprintf(fid,'%f %f %f %f %d\n',bbs(ind(j),1),bbs(ind(j),2),bbs(ind(j),3),bbs(ind(j),4),r.class(ind(j)));
    end
    fclose(fid);
end