function rank_by_feature_pair(f1, f2, model, dataset, year)

% take a model
% select highest scoring features

TOP_K = 1000;
NUM_FEAT = 20;

conf = voc_config('pascal.year', year);
VOCopts = conf.pascal.VOCopts;

[ids, cls_label] = textread(sprintf(VOCopts.imgsetpath, [model.class '_' dataset]), '%s %d');
P = find(cls_label == 1);
N = find(cls_label == -1);
cls_ids = ids(P);
not_cls_ids = ids(N);
ids = textread(sprintf(VOCopts.imgsetpath, dataset), '%s');
ids = cat(1, not_cls_ids(1:min(length(not_cls_ids), 2*length(cls_ids))), cls_ids);

% value, ids ind, box ind
top_boxes = zeros(0, 3+4);

opts.layer = 'pool5';
opts.fine_tuned = 1;
opts.use_flipped = 0;

for i = 1:length(ids)
  tic_toc_print('%d/%d\n', i, length(ids));
  d = load_cached_features(dataset, year, ids{i}, opts);

  bs = [d.boxes d.feat(:,f1)+d.feat(:,f2)];
  sel = fast_nms(bs, 0.1);
  sz = length(sel);
  % score, image_ids_index, ignore, box
  new_boxes = [d.feat(sel,f1)+d.feat(sel,f2) ones(sz,1)*i (1:sz)' d.boxes(sel,:)];

  top_boxes = cat(1, top_boxes, new_boxes);
  [~, ord] = sort(top_boxes(:,1), 'descend');
  if length(ord) > TOP_K 
    ord = ord(1:TOP_K);
  end
  top_boxes = top_boxes(ord,:);
end

save_file = sprintf('paper-figures/pair_%d_%d_rank_by_feature_ft_%d', ...
                    f1, f2, opts.fine_tuned);
save(save_file, 'f1', 'f2', 'ids', 'top_boxes', 'VOCopts', 'model');
