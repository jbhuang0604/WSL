function f = xform_feat_custom(f, opts)

%f = f.^pwr;
%f = bsxfun(@times, f, 1./sqrt(sum(f.^2, 2)));

%target_norm = 20;

target_norm = opts.target_norm;

f = f .* (target_norm / opts.feat_norm_mean);
