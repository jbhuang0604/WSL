function cache_fc8_features(imdb, varargin)

ip = inputParser;
ip.addRequired('imdb', @isstruct);
ip.addOptional('start', 1, @isscalar);
ip.addOptional('end', 0, @isscalar);
ip.addOptional('crop_mode', 'warp', @isstr);
ip.addOptional('crop_padding', 16, @isscalar);
ip.addOptional('net_file', '', @isstr);
ip.addOptional('cache_name', '', @isstr);

ip.parse(imdb, varargin{:});
opts = ip.Results;
opts.net_def_file = './prototxt/caffenet_fc8.prototxt';
opts.batch_size = 50;
opts.crop_size = 227;

image_ids = imdb.image_ids;
if opts.end == 0
  opts.end = length(image_ids);
end

% Where to save feature cache
if ~exist('cache','file')
    mkdir('cache');
end
opts.output_dir = ['./cache/' opts.cache_name '/'];
mkdir_if_missing(opts.output_dir);

fprintf('\n\n~~~~~~~~~~~~~~~~~~~~~~~~~~~~\n');
fprintf('Feature caching options:\n');
disp(opts);
fprintf('~~~~~~~~~~~~~~~~~~~~~~~~~~~~\n\n');

% load the region of interest database
roidb = imdb.roidb_func(imdb);

if caffe('is_initialized') == 0
    caffe('init', opts.net_def_file, opts.net_file, 'test');
end
% caffe('set_mode_cpu');
caffe('set_mode_gpu');
% caffe('set_device',3);

total_time = 0;
count = 0;
for i = opts.start:opts.end
  fprintf('%s: cache features: %d/%d\n', procid(), i, opts.end);
  save_file = [opts.output_dir image_ids{i} '.mat'];
  if exist(save_file, 'file') ~= 0
    fprintf(' [already exists]\n');
    continue;
  end
  count = count + 1;
  tot_th = tic;
  d = roidb.rois(i);
  im = imread(imdb.image_at(i));
  if size(im,3)~=3
      im = cat(3,im,im,im);
  end
  th = tic;
  d.feat_in = cache_bb_features(im, d.boxes, opts, 1);
  d.feat_out = cache_bb_features(im, d.boxes, opts, 2);
  fprintf(' [features: %.3fs]\n', toc(th));
  th = tic;
  save(save_file, '-struct', 'd');
  fprintf(' [saving:   %.3fs]\n', toc(th));
  total_time = total_time + toc(tot_th);
  fprintf(' [avg time: %.3fs (total: %.3fs)]\n', ...
      total_time/count, total_time);
end
