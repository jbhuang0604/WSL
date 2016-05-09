function feature_pairs = pick_feature_pair(model, N)

ovM = get_overlap_matrix();
feature_pairs = zeros(0, 2);

[~, w_ord] = sort(model.w, 'descend');

for i = 1:N
    u1 = w_ord(i);
    pos1 = mod(u1-1, 36)+1;
    ov = ovM(pos1,:);
    ok = repmat((ov < 1/3), [1 256]);
    ok(u1-pos1+1:u1-pos1+36) = 0;
    w = model.w;
    w(~ok) = -inf;
    [~, ord] = sort(w, 'descend');
    u2 = ord(1);
    feature_pairs = cat(1, feature_pairs, [u1 u2]);
end



function ovM = get_overlap_matrix()

ovM = zeros(36);

s = 224/6;
points = round(s/2:s:224);

for i = 1:36
  M = zeros(6,6,256);
  M(i) = 1;
  M = sum(M, 3)';
  [r,c] = find(M);
  r1_1 = max(1, points(r) - 81);
  r1_2 = min(224, points(r) + 81);
  c1_1 = max(1, points(c) - 81);
  c1_2 = min(224, points(c) + 81);

  for j = 1:36
    M = zeros(6,6,256);
    M(j) = 1;
    M = sum(M, 3)';
    [r,c] = find(M);
    r2_1 = max(1, points(r) - 81);
    r2_2 = min(224, points(r) + 81);
    c2_1 = max(1, points(c) - 81);
    c2_2 = min(224, points(c) + 81);

    ovM(i,j) = boxoverlap([c1_1 r1_1 c1_2 r1_2], [c2_1 r2_1 c2_2 r2_2]);
  end
end
