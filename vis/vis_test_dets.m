function vis_test_dets(split, year, cls)

conf = voc_config('pascal.year', year);

VOCopts  = conf.pascal.VOCopts;

res_fn = sprintf(VOCopts.detrespath, 'comp4', cls);

[ids, scores, x1, y1, x2, y2] = textread(res_fn, '%s %f %f %f %f %f');

[~, ord] = sort(scores, 'descend');

for i = 1:length(ord)
  j = ord(i);

  im = imread(sprintf(VOCopts.imgpath, ids{j})); 

  showboxes(im, [x1(j) y1(j) x2(j) y2(j)]);
  title(num2str(scores(j)));
  pause;
end
