function aboxes = rcnn_test(rcnn_model, imdb, suffix)
% res = rcnn_test(rcnn_model, imdb, suffix)
%   Compute test results using the trained rcnn_model on the
%   image database specified by imdb. Results are saved
%   with an optional suffix.

% returns aboxes, a two-dimensional cell array
% with the first dimension being classes, the second
% dimension images. 

% AUTORIGHTS
% ---------------------------------------------------------
% Copyright (c) 2014, Ross Girshick
% 
% This file is part of the R-CNN code and is available 
% under the terms of the Simplified BSD License provided in 
% LICENSE. Please retain this notice and LICENSE if you use 
% this file (or any portion of it) in your project.
% ---------------------------------------------------------

conf = rcnn_config('sub_dir', imdb.name);
image_ids = imdb.image_ids;

% assume they are all the same
feat_opts.layer = 8; % go to layer 8
feat_opts.cache_name = 'finetune_kitti_iter_41000';
num_classes = imdb.num_classes;

if ~exist('suffix', 'var') || isempty(suffix)
  suffix = '';
else
  suffix = ['_' suffix];
end

try
  aboxes = cell(num_classes, 1);
  for i = 1:num_classes
    load([conf.cache_dir rcnn_model.classes{i} '_boxes_' imdb.name suffix]);
    aboxes{i} = boxes;
  end
catch
  aboxes = cell(num_classes, 1);
  box_inds = cell(num_classes, 1);
  for i = 1:num_classes
    aboxes{i} = cell(length(image_ids), 1);
    box_inds{i} = cell(length(image_ids), 1);
  end

  % heuristic that yields at most 100k pre-NMS boxes
  % per 2500 images
  max_per_set = ceil(100000/2500)*length(image_ids);
  max_per_image = 100;
  top_scores = cell(num_classes, 1);
  thresh = -inf(num_classes, 1);
  box_counts = zeros(num_classes, 1);

  if ~isfield(rcnn_model, 'folds')
    folds{1} = 1:length(image_ids);
  else
    folds = rcnn_model.folds;
  end

  count = 0;
  for f = 1:length(folds)
    for i = folds{f}
      count = count + 1;
      fprintf('%s: test (%s) %d/%d\n', procid(), imdb.name, count, length(image_ids));
      d = rcnn_load_cached_pool5_features(feat_opts.cache_name, ...
          imdb.name, image_ids{i});
      if isempty(d.feat)
        fprintf('Could not load pool5 features!');
        continue;
      end
      d.feat = rcnn_pool5_to_fcX(d.feat, feat_opts.layer, rcnn_model);
      % to each box, assign the class of the largest score
      % (1 = background, 2 = first foreground object, etc)
      [score,classes] = max(d.feat, [], 2);
      for j = 1:num_classes
        boxes = d.boxes;
        I = find(classes == j+1 & score > thresh(j));
        boxes = boxes(I,:);
        scores = score(I);
        aboxes{j}{i} = cat(2, single(boxes), single(scores));
        [~, ord] = sort(scores, 'descend');
        ord = ord(1:min(length(ord), max_per_image));
        aboxes{j}{i} = aboxes{j}{i}(ord, :);
        box_inds{j}{i} = I(ord);

        box_counts(j) = box_counts(j) + length(ord);
        top_scores{j} = cat(1, top_scores{j}, scores(ord));
        top_scores{j} = sort(top_scores{j}, 'descend');
        if box_counts(j) > max_per_set
          top_scores{j}(max_per_set+1:end) = [];
          thresh(j) = top_scores{j}(end);
        end
      end
    end
  end

  for i = 1:num_classes
    % go back through and prune out detections below the found threshold
    for j = 1:length(image_ids)
      if ~isempty(aboxes{i}{j})
        I = find(aboxes{i}{j}(:,end) < thresh(i));
        aboxes{i}{j}(I,:) = [];
        box_inds{i}{j}(I,:) = [];
        
        % perform nms
        keep = nms(aboxes{i}{j}, 0.3);
        aboxes{i}{j} = aboxes{i}{j}(keep,:);
      end
    end

    save_file = [conf.cache_dir rcnn_model.classes{i} '_boxes_' imdb.name suffix];
    boxes = aboxes{i};
    inds = box_inds{i};
    save(save_file, 'boxes', 'inds');
    clear boxes inds;
  end
end