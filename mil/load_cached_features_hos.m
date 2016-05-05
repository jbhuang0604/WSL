function d = load_cached_features_hos(dataset, year, ids)

base_path = ['../cache/voc_' year '_' dataset '_fc7'];
file = sprintf('%s/%s.mat', base_path, ids);
d = load(file);