function show_latent_choice(model, trainset, year)

conf = voc_config('pascal.year', year);
dataset.year = year;
dataset.trainset = trainset;
dataset.image_ids = textread(sprintf(conf.pascal.VOCopts.imgsetpath, trainset), '%s');

[ids, cls_label] = textread(sprintf(conf.pascal.VOCopts.imgsetpath, [model.class '_' trainset]), '%s %d');
P = find(cls_label == 1);
dataset.image_ids = ids(P);

get_positive_features(model, dataset, conf);

% ------------------------------------------------------------------------
function get_positive_features(model, dataset, conf)
% ------------------------------------------------------------------------

thresh = 0.7;

for i = 1:length(dataset.image_ids)
  tic_toc_print('%s: pos features %d/%d\n', ...
                procid(), i, length(dataset.image_ids));
  d = load_cached_features(dataset.trainset, dataset.year, dataset.image_ids{i}, model.opts);
  d.feat = xform_feat(d.feat, model.opts);
  im = imread(sprintf(conf.pascal.VOCopts.imgpath, dataset.image_ids{i})); 
  % boxes who overlap a gt by > 70%
  I = find(d.overlap(:,model.class_id) > thresh);
  zs = d.feat(I,:)*model.w + model.b;
  I_gt = find(d.class == model.class_id);
  for k = 1:length(I_gt)
    ovr = boxoverlap(d.boxes(I,:), d.boxes(I_gt(k),:));
    %I_ovr = find((ovr > thresh) & (ovr ~= 1));
    I_ovr = find(ovr > thresh);
    [~, argmax] = max(zs(I_ovr));
    sel = I(I_ovr(argmax));
    showboxesc(im, d.boxes(I_gt(k), :), 'g', '-');
    showboxesc([], d.boxes(sel, :), 'r', '--');
    title(sprintf('%.3f', zs(I_ovr(argmax))));
    pause;
  end
end
