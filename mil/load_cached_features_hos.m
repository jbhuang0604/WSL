function d = load_cached_features_hos(select, dataset, year, ids)

if select == 1
    base_path = ['../cache/voc_' year '_' dataset '_fc7/mil_train'];
    file = sprintf('%s/%s.mat', base_path, ids);
else
    base_path = ['../cache/voc_' year '_' dataset '_fc7/mil_test'];
    file = sprintf('%s/%s.mat', base_path, ids);
end
d = load(file);
