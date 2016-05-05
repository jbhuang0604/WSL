function showboxes_hos(im, boxes, color, draw, cid, pid)

if nargin <= 3
  draw = true;
end

%optionally print cid
if nargin <= 4
  cid = nan; 
end

if nargin == 5
  pid = nan;
end

% doing this initializes the image. set to false in consequent runs. 
% if this function is to be called multiple times on the same image.
if draw
  image(im);
  axis image;
  axis off;
end

if ~isempty(boxes)
  for j = 1:size(boxes,1)
    
    x1 = boxes(j,1);
    y1 = boxes(j,2);
    x2 = boxes(j,3);
    y2 = boxes(j,4);
    
    line([x1 x1 x2 x2 x1 x1]', [y1 y2 y2 y1 y1 y2]', 'color', color, ...
                                                       'linewidth', 1);
                                                     
    if isnan(cid)                                                 
      text(x1+3, y1+17, num2str(j), 'backgroundcolor', color);  
    elseif isnan(pid)
      text(x1+3, y1+17, ['C', num2str(cid), '_', num2str(j)],...
        'backgroundcolor', color, 'fontsize', 5);  
    else
      text(x1+3, y1+17, ['C', num2str(cid), '_', num2str(pid)],...
        'backgroundcolor', color, 'fontsize', 5);  
    end
    
  end
end
drawnow;