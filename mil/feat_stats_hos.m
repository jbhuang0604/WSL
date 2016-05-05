function [mean_norm, stdd] = feat_stats_hos(split, year, layer, fine_tuned)

num_images = 200;
boxes_per_image = 200;

conf = voc_config('pascal.year', year);

image_ids = textread(sprintf(conf.pascal.VOCopts.imgsetpath, split), '%s');
image_ids = image_ids(randperm(length(image_ids), num_images));

feat_opts.layer = layer;
feat_opts.fine_tuned = fine_tuned;
feat_opts.use_flipped = false;

ns = [];
for i = 1:length(image_ids)
  tic_toc_print('feat stats: %d/%d\n', i, length(image_ids));
  d = load_cached_features_hos(split, year, image_ids{i});
  %d.feat = xform_feat(d.feat, pwr);
  %d.feat = d.feat ./ 19.584 * 20;

  X = d.feat(randperm(size(d.feat,1), min(boxes_per_image, size(d.feat,1))), :);
  ns = cat(1, ns, sqrt(sum(X.^2, 2)));
end

mean_norm = mean(ns);
stdd = std(ns);
