function im_out = extract_region_out(im, bbox)

im_out = im;
bbox(1) = max(1, bbox(1));
bbox(2) = max(1, bbox(2));
bbox(3) = min(size(im,2), bbox(3));
bbox(4) = min(size(im,1), bbox(4));
im_out(bbox(2):bbox(4),bbox(1):bbox(3),1) = 123;
im_out(bbox(2):bbox(4),bbox(1):bbox(3),2) = 117;
im_out(bbox(2):bbox(4),bbox(1):bbox(3),3) = 104;
im_out = preprocess(im_out);