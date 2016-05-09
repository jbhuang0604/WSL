function rank_by_feature(model, dataset, year)

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
%ids = cls_ids(1:10);

% select features
sel_features = find(model.w > 0);
[~, ord] = sort(model.w(sel_features), 'descend');
sel_features = sel_features(ord(1:NUM_FEAT))';
% ---- sanity check ----
vals = sort(model.w, 'descend');
assert(vals(1) == max(model.w));
assert(all(vals(1:NUM_FEAT) == model.w(sel_features)));
clear vals;
% ----------------------
% add most negative features, too
sel_features_neg = find(model.w < 0);
[~, ord] = sort(model.w(sel_features_neg));
sel_features = cat(2, sel_features, sel_features_neg(ord(1:NUM_FEAT))');

if 0
  % rand features
  sel_features = randperm(length(model.w), NUM_FEAT);
end

disp(model.w(sel_features));

% value, ids ind, box ind
top_boxes = {};
for i = sel_features
  top_boxes{i} = zeros(0, 3+4);
end

overlap = [];

for i = 1:length(ids)
  tic_toc_print('%d/%d\n', i, length(ids));
  d = load_cached_features(dataset, year, ids{i}, model.opts);
  overlap = cat(1, overlap, d.overlap(:, model.class_id));
  if 0
    % only take high overlap boxes
    selo = find(d.overlap(:, model.class_id) > 0.8);
    d.feat = d.feat(selo,:);
    d.boxes = d.boxes(selo,:);
  end
  for f = sel_features
    bs = [d.boxes d.feat(:,f)];
    sel = fast_nms(bs, 0.1);
    %sel2 = nms(bs, 0.1);
    %assert(length(intersect(sel, sel2)) == length(sel));
    sz = length(sel);
    % score, image_ids_index, ignore, box
    new_boxes = [d.feat(sel,f) ones(sz,1)*i (1:sz)' d.boxes(sel,:)];

    top_boxes{f} = cat(1, top_boxes{f}, new_boxes);
    [~, ord] = sort(top_boxes{f}(:,1), 'descend');
    if length(ord) > TOP_K 
      ord = ord(1:TOP_K);
    end
    top_boxes{f} = top_boxes{f}(ord,:);
  end
end

save_file = sprintf('paper-figures/%s_rank_by_feature_ft_%d', ...
                    model.class, model.opts.fine_tuned);
save(save_file, 'sel_features', 'ids', ...
     'top_boxes', 'overlap', 'VOCopts', 'model');
