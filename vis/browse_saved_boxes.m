function browse_saved_boxes(dataset, year)

conf = voc_config('pascal.year', year);
VOCopts = conf.pascal.VOCopts;

ids = textread(sprintf(VOCopts.imgsetpath, dataset), '%s');

for i = 1:length(ids)
  d = load_cached_features(dataset, year, ids{i});
  im = imread(sprintf(VOCopts.imgpath, ids{i})); 
  for j = 1:size(d.boxes,1)
    showboxes(im, d.boxes(j, :));
    title(sprintf('%d %d', d.gt(j), d.class(j)));
    pause;
  end
end
