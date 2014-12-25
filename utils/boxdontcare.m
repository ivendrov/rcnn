function dontcare = boxdontcare(a,bs)
%BOXDONTCARE returns a boolean vector
% indicating which boxes that are rows of a
% overlap with a "don't care" box (a row of bs)

% currently an "overlap" happens if the intersection over 
% the area of either box is over 0.5
[numBoxes,~] = size(a);
[numDontCares,~] = size(bs);
dontcare = false(numBoxes,1);
for i = 1:numDontCares
    b = bs(i,:);

    x1 = max(a(:,1), b(1));
    y1 = max(a(:,2), b(2));
    x2 = min(a(:,3), b(3));
    y2 = min(a(:,4), b(4));

    w = x2-x1+1;
    h = y2-y1+1;
    inter = w.*h;
    aarea = (a(:,3)-a(:,1)+1) .* (a(:,4)-a(:,2)+1);
    barea = (b(3)-b(1)+1) * (b(4)-b(2)+1);

    dontcarei = (max(inter ./ aarea, inter ./barea) >= 0.5);

    % set invalid entries to not overlap
    dontcarei(w <= 0) = false;
    dontcarei(h <= 0) = false;
    
    dontcare = dontcare | dontcarei;
end

end

