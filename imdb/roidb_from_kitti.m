function roidb = roidb_from_kitti(imdb)
% roidb = roidb_from_voc(imdb)
%   Builds an regions of interest database from imdb image
%   database. Uses precomputed selective search boxes available
%   in the R-CNN data package. (Adapted by Ivan Vendrov to work with KITTI)
%
%   Inspired by Andrea Vedaldi's MKL imdb and roidb code.

% AUTORIGHTS
% ---------------------------------------------------------
% Copyright (c) 2014, Ross Girshick
% 
% This file is part of the R-CNN code and is available 
% under the terms of the Simplified BSD License provided in 
% LICENSE. Please retain this notice and LICENSE if you use 
% this file (or any portion of it) in your project.
% ---------------------------------------------------------

cache_file = ['./imdb/cache/roidb_' imdb.name];
try
  load(cache_file);
catch
  roidb.name = imdb.name;

  fprintf('Loading region proposals...');
  regions_file = sprintf('./data/selective_search_data/%s', roidb.name);
  load(regions_file);
  fprintf('done\n');

  for i = 1:length(imdb.image_ids)
    tic_toc_print('roidb (%s): %d/%d\n', roidb.name, i, length(imdb.image_ids));
    
    % TODO GET GROUND TRUTH OBJECTS; objects is a struct array with fields
    % bbox and class.
    % get image #
    [~,number,~] = fileparts(imdb.image_at(i));
    labels = readLabels('~/kitti/object_detection/training/label_2', round(str2double(number)));
    
    % read the objects and dont cares from the label file
    nObjects = 0;
    nDontCares = 0;
    dontcares = [];
    for j = 1:length(labels);
        l = labels(j);
        bbox = [l.x1 l.y1 l.x2 l.y2];
        if (strcmpi(l.type, 'dontcare'))
            nDontCares = nDontCares + 1;
            dontcares(nDontCares,:) = bbox;
        elseif (isKey(imdb.class_to_id, l.type))
            nObjects = nObjects + 1;
            objects(nObjects).class = l.type;
            objects(nObjects).bbox = [l.x1 l.y1 l.x2 l.y2];
        else
            fprintf('unknown annotation: %s\n', l.type);
        end
    end
    
    % remove all regions which overlap with the dont care boxes 
    boxes = regions.boxes{i};
    overlaps = boxdontcare(boxes(:,[2 1 4 3]), dontcares);
    fprintf('removed %d dontcare regions\n', sum(overlaps));
    newBoxes = boxes(~overlaps,:);
    regions.boxes{i} = newBoxes;
    
    roidb.rois(i) = attach_proposals(objects, regions.boxes{i}, imdb.class_to_id);
  end

  fprintf('Saving roidb to cache...');
  save(cache_file, 'roidb', '-v7.3');
  save([regions_file '_pruned'], 'regions'); % only save the regions that don't overlap with dont cares.
  fprintf('done\n');
end


% ------------------------------------------------------------------------
function rec = attach_proposals(objects, boxes, class_to_id)
% ------------------------------------------------------------------------

% change selective search order from [y1 x1 y2 x2] to [x1 y1 x2 y2]
boxes = boxes(:, [2 1 4 3]);

%           gt: [2108x1 double]
%      overlap: [2108x20 single]
%      dataset: 'voc_2007_trainval'
%        boxes: [2108x4 single]
%         feat: [2108x9216 single]
%        class: [2108x1 uint8]
gt_boxes = cat(1, objects(:).bbox);
all_boxes = cat(1, gt_boxes, boxes);
gt_classes = class_to_id.values({objects(:).class});
gt_classes = cat(1, gt_classes{:});
num_gt_boxes = size(gt_boxes, 1);

num_boxes = size(boxes, 1);

rec.gt = cat(1, true(num_gt_boxes, 1), false(num_boxes, 1));
rec.overlap = zeros(num_gt_boxes+num_boxes, class_to_id.Count, 'single');
for i = 1:num_gt_boxes
  rec.overlap(:, gt_classes(i)) = ...
      max(rec.overlap(:, gt_classes(i)), boxoverlap(all_boxes, gt_boxes(i, :)));
end
rec.boxes = single(all_boxes);
rec.feat = [];
rec.class = uint8(cat(1, gt_classes, zeros(num_boxes, 1)));
