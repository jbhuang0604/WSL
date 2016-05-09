function show_detections(model, split, year)

conf = voc_config('pascal.year', year);
dataset.year = year;
dataset.trainset = split;
dataset.image_ids = textread(sprintf(conf.pascal.VOCopts.imgsetpath, split), '%s');
show_det(model, dataset, conf);

% ------------------------------------------------------------------------
function show_det(model, dataset, conf)
% ------------------------------------------------------------------------

for i = 1:length(dataset.image_ids)
  tic_toc_print('%s: %d/%d\n', ...
                procid(), i, length(dataset.image_ids));
  d = load_cached_features_hos(dataset.trainset, dataset.year, dataset.image_ids{i}, model.opts);

  if isempty(find(d.class == model.class_id))
    continue;
  end

  im = imread(sprintf(conf.pascal.VOCopts.imgpath, dataset.image_ids{i})); 
  % boxes who overlap a gt by > 70%
  z = d.feat*model.w + model.b;

  I = find(~d.gt & z > -1);
  boxes = cat(2, single(d.boxes(I,:)), z(I));
  [~, ord] = sort(z(I), 'descend');
  ord = ord(1:min(length(ord), 20));
  boxes = boxes(ord, :);

%  nms_interactive(im, boxes, 0.3);

%  keep = 1:size(boxes,1); 
  keep = nms(boxes, 0.3);
  showboxes(im, boxes(keep,1:4));
  pause;
%   for k = 1:length(keep)
%     showboxes(im, boxes(keep(k),1:4));
%     title(sprintf('score: %.3f\n', boxes(keep(k),end)));
%     pause;
%   end
end
