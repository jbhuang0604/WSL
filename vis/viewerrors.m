function ap = viewerrors(model, boxes, testset, year, saveim)
% For visualizing mistakes on a validation set

% AUTORIGHTS
% -------------------------------------------------------
% Copyright (C) 2009-2012 Ross Girshick
% 
% This file is part of the voc-releaseX code
% (http://people.cs.uchicago.edu/~rbg/latent/)
% and is available under the terms of an MIT-like license
% provided in COPYING. Please retain this notice and
% COPYING if you use this file (or a portion of it) in
% your project.
% -------------------------------------------------------

SHOW_TP = true;
SHOW_FP = true;
SHOW_FN = true;
im_path = sprintf('~/public_html/private/convnet-sel-search/%s/', model.class);
if exist(im_path) == 0
  unix(['mkdir -p ' im_path]);
end
fp_html = [im_path 'fp.html'];
fn_html = [im_path 'fn.html'];
tp_html = [im_path 'tp.html'];
index_html = [im_path 'index.html'];

if saveim
  htmlfid = fopen(index_html, 'w');
  fprintf(htmlfid, '<html><body>');
  fprintf(htmlfid, '<h2>%s</h2>', model.class);
  fprintf(htmlfid, '<a href="tp.html">true positives</a><br />');
  fprintf(htmlfid, '<a href="fp.html">false positives</a><br />');
  fprintf(htmlfid, '<a href="fn.html">missed</a><br />');
  fprintf(htmlfid, '</html></body>');
end


warning on verbose;
warning off MATLAB:HandleGraphics:noJVM;

cls = model.class;

conf = voc_config('pascal.year', year, ...
                  'eval.test_set', testset);
VOCopts  = conf.pascal.VOCopts;
cachedir = conf.paths.model_dir;

% Load test set ground-truth
fprintf('%s: viewerrors: loading ground truth\n', cls);
[gtids, recs, hash, gt, npos] = load_ground_truth(cls, conf);

% Load detections from the model
[ids, confidence, BB] = get_detections(boxes, cls, conf);

% sort detections by decreasing confidence
[sc, si] = sort(-confidence);
ids = ids(si);
BB = BB(:,si);

% assign detections to ground truth objects
nd = length(confidence);
tp = zeros(nd,1);
fp = zeros(nd,1);
md = zeros(nd,1);
od = zeros(nd,1);
jm = zeros(nd,1);
for d = 1:nd
  % display progress
  tic_toc_print('%s: pr: compute: %d/%d\n', cls, d, nd);
  
  % find ground truth image
  i = xVOChash_lookup(hash, ids{d});
  if isempty(i)
    error('unrecognized image "%s"', ids{d});
  elseif length(i) > 1
    error('multiple image "%s"', ids{d});
  end

  % assign detection to ground truth object if any
  % reported detection
  bb = BB(:,d);
  ovmax = -inf;
  jmax = 0;
  % loop over bounding boxes for this class in the gt image
  for j = 1:size(gt(i).BB,2)
    % consider j-th gt box
    bbgt = gt(i).BB(:,j);
    % compute intersection box
    bi = [max(bb(1), bbgt(1)); ...
          max(bb(2), bbgt(2)); ...
          min(bb(3), bbgt(3)); ...
          min(bb(4), bbgt(4))];
    iw = bi(3)-bi(1)+1;
    ih = bi(4)-bi(2)+1;
    if iw > 0 & ih > 0                
      % compute overlap as area of intersection / area of union
      ua = (bb(3)-bb(1)+1) * (bb(4)-bb(2)+1) + ...
           (bbgt(3)-bbgt(1)+1) * (bbgt(4)-bbgt(2)+1) - ...
           iw * ih;
      ov = iw * ih / ua;
      if ov > ovmax
        ovmax = ov;
        jmax = j;
      end
    end
  end
  % assign detection as true positive/don't care/false positive
  if jmax > 0 && ovmax > gt(i).overlap(jmax)
    gt(i).overlap(jmax) = ovmax;
    gt(i).best_boxes(jmax,:) = bb';
  end
  od(d) = ovmax;
  jm(d) = jmax;
  if ovmax >= VOCopts.minoverlap
    if ~gt(i).diff(jmax)
      if ~gt(i).det(jmax)
        % true positive
        tp(d) = 1;
        gt(i).det(jmax) = true;
        gt(i).tp_boxes(jmax,:) = bb';
      else
        % false positive (multiple detection)
        fp(d) = 1;
        md(d) = 1;
      end
    end
  else
    % false positive (low or no overlap)
    fp(d) = 1;
  end
end

% compute precision/recall
cfp = cumsum(fp);
ctp = cumsum(tp);
rec = ctp/npos;
prec = ctp./(cfp+ctp);

fprintf('total recalled = %d/%d (%.1f%%)\n', sum(tp), npos, 100*sum(tp)/npos);

if SHOW_TP
  if saveim
    htmlfid = fopen(tp_html, 'w');
    fprintf(htmlfid, '<html><body>');
  end

  fprintf('displaying true positives\n');
  count = 0;
  d = 1;
  while d < nd && count < 400
    if tp(d)
      count = count + 1;
      i = xVOChash_lookup(hash, ids{d});
      im = imread([VOCopts.datadir recs(i).imgname]);

      % Recompute the detection to get the derivation tree
      score = -sc(d);

      subplot(1,2,1);
      imagesc(im);
      axis image;
      axis off;

      subplot(1,2,2);
      showboxesc(im, BB(:,d)', 'r', '-');

      str = sprintf('%d det# %d/%d: @prec: %0.3f  @rec: %0.3f\nscore: %0.3f  GT overlap: %0.3f', count, d, nd, prec(d), rec(d), -sc(d), od(d));

      fprintf('%s', str);
      title(str);

      fprintf('\n');

      if saveim
        cmd = sprintf('export_fig %s/%s-%d-tp.jpg -jpg', im_path, cls, d);
        eval(cmd);
        fprintf(htmlfid, sprintf('<img src="%s-%d-tp.jpg" />\n', cls, d));
        fprintf(htmlfid, '<br /><br />\n');
      else
        pause;
      end
    end
    d = d + 1;
  end

  if saveim
    fprintf(htmlfid, '</body></html>');
    fclose(htmlfid);
  end
end


if SHOW_FP
  if saveim
    htmlfid = fopen(fp_html, 'w');
    fprintf(htmlfid, '<html><body>');
  end

  fprintf('displaying false positives\n');
  count = 0;
  d = 1;
  while d < nd && count < 400
    if fp(d)
      count = count + 1;
      i = xVOChash_lookup(hash, ids{d});
      im = imread([VOCopts.datadir recs(i).imgname]);

      % Recompute the detection to get the derivation tree
      score = -sc(d);

      subplot(1,2,1);
      imagesc(im);
      axis image;
      axis off;

      subplot(1,2,2);
      showboxesc(im, BB(:,d)', 'r', '-');

      str = sprintf('%d det# %d/%d: @prec: %0.3f  @rec: %0.3f\nscore: %0.3f  GT overlap: %0.3f', count, d, nd, prec(d), rec(d), -sc(d), od(d));
      if md(d)
        str = sprintf('%s mult det', str);
      end
      if fp(d) && jm(d) > 0
        str = sprintf('%s\nmax overlap all det: %0.3f', str, gt(i).overlap(jm(d)));
      end

      fprintf('%s', str);
      title(str);

      fprintf('\n');

      if saveim
        cmd = sprintf('export_fig %s/%s-%d-fp.jpg -jpg', im_path, cls, d);
        eval(cmd);
        fprintf(htmlfid, sprintf('<img src="%s-%d-fp.jpg" />\n', cls, d));
        fprintf(htmlfid, '<br /><br />\n');
      else
        pause;
      end
    end
    d = d + 1;
  end

  if saveim
    fprintf(htmlfid, '</body></html>');
    fclose(htmlfid);
  end
end

if SHOW_FN
  % to find false negatives loop over gt(i) and display any box that has
  % gt(i).det(j) == false && ~gt(i).diff(j)
  fprintf('displaying false negatives\n');

  if saveim
    htmlfid = fopen(fn_html, 'w');
    fprintf(htmlfid, '<html><body>');
  end

  clf;

  count = 0;
  for i = 1:length(gt)
    if count >= 200
      break;
    end
    s = 0;
    if ~isempty(gt(i).det)
      s = sum((~gt(i).diff)' .* (~gt(i).det));
    end
    if s > 0
      diff = [];
      fn = [];
      tp = [];
      best_boxes = [];
      best_ovrs = [];
      fprintf('%d\n', i);
      [gt(i).diff(:) gt(i).det(:) gt(i).overlap(:)]
      for j = 1:length(gt(i).det)
        bbgt = gt(i).BB(:,j)';
        if gt(i).diff(j)
          diff = [diff; [bbgt 0]];
        elseif ~gt(i).det(j)
          fn = [fn; [bbgt 1]];
          best_boxes = cat(1, best_boxes, gt(i).best_boxes(j,:));
          best_ovrs = cat(1, best_ovrs, gt(i).overlap(j));
        else
          tp = [tp; [bbgt 2]];
          tp = [tp; [gt(i).tp_boxes(j,:) 3]];
        end
      end
      im = imread([VOCopts.datadir recs(i).imgname]);
      showboxesc(im, [diff; fn; tp]);
      for j = 1:length(best_ovrs)
        if best_ovrs(j) > -inf
          showboxesc([], best_boxes(j,:), 'y', '--');
          text(best_boxes(j,1), best_boxes(j,2), sprintf('%0.3f', best_ovrs(j)), 'BackgroundColor', [.7 .9 .7]);
        end
      end

      if saveim
        cmd = sprintf('export_fig %s/%s-%d-fn.jpg -jpg', im_path, cls, count);
        eval(cmd);
        fprintf(htmlfid, sprintf('<img src="%s-%d-fn.jpg" />\n', cls, count));
        fprintf(htmlfid, '<br /><br />\n');
      else
        pause;
      end;
      count = count + 1;
    end
  end

  if saveim
    fprintf(htmlfid, '</body></html>');
    fclose(htmlfid);
  end
end


function [gtids, recs, hash, gt, npos] = load_ground_truth(cls, conf)

VOCopts  = conf.pascal.VOCopts;
year     = conf.pascal.year;
cachedir = conf.paths.model_dir;
testset  = conf.eval.test_set;

cp = [cachedir cls '_ground_truth_' testset '_' year];
try
  load(cp, 'gtids', 'recs', 'hash', 'gt', 'npos');
catch
  [gtids, t] = textread(sprintf(VOCopts.imgsetpath,VOCopts.testset), '%s %d');
  for i = 1:length(gtids)
    % display progress
    tic_toc_print('%s: pr: load: %d/%d\n', cls, i, length(gtids));
    % read annotation
    recs(i) = PASreadrecord(sprintf(VOCopts.annopath, gtids{i}));
  end

  % hash image ids
  hash = xVOChash_init(gtids);
     
  % extract ground truth objects
  npos = 0;
  gt(length(gtids)) = struct('BB', [], 'diff', [], 'det', [], 'overlap', [], 'tp_boxes', []);
  for i = 1:length(gtids)
    % extract objects of class
    clsinds = strmatch(cls, {recs(i).objects(:).class}, 'exact');
    gt(i).BB = cat(1, recs(i).objects(clsinds).bbox)';
    gt(i).diff = [recs(i).objects(clsinds).difficult];
    gt(i).det = false(length(clsinds), 1);
    gt(i).overlap = -inf*ones(length(clsinds), 1);
    gt(i).tp_boxes = zeros(length(clsinds), 4);
    gt(i).best_boxes = zeros(length(clsinds), 4);
    npos = npos + sum(~gt(i).diff);
  end

  save(cp, 'gtids', 'recs', 'hash', 'gt', 'npos');
end



function [ids, confidence, BB] = get_detections(boxes, cls, conf)

VOCopts  = conf.pascal.VOCopts;
year     = conf.pascal.year;
cachedir = conf.paths.model_dir;
testset  = conf.eval.test_set;

ids = textread(sprintf(VOCopts.imgsetpath, testset), '%s');

% Write and read detection data in the same way as pascal_eval.m 
% and the VOCdevkit

% write out detections in PASCAL format and score
fid = fopen(sprintf(VOCopts.detrespath, 'comp3', cls), 'w');
for i = 1:length(ids);
  bbox = boxes{i};
  keep = nms(bbox, 0.3);
  bbox = bbox(keep,:);
  for j = 1:size(bbox,1)
    fprintf(fid, '%s %.14f %d %d %d %d\n', ids{i}, bbox(j,end), bbox(j,1:4));
  end
end
fclose(fid);
[ids, confidence, b1, b2, b3, b4] = ...
  textread(sprintf(VOCopts.detrespath, 'comp3', cls), '%s %f %f %f %f %f');
BB = [b1 b2 b3 b4]';
