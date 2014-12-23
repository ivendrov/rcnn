function imdb = imdb_from_kitti(image_set)
% builds an image database from the selected KITTI dataset (train or val),
% TODO add handling for test images
root_dir = '~/kitti/object_detection/training/image_2/';
image_dir = fullfile(root_dir, image_set); % image_set should be train or val

% ---------------------------------------------------------
% Copyright (c) 2014, Ross Girshick
% 
% This file is part of the R-CNN code and is available 
% under the terms of the Simplified BSD License provided in 
% LICENSE. Please retain this notice and LICENSE if you use 
% this file (or any portion of it) in your project.
% ---------------------------------------------------------

%imdb.name = 'voc_train_2007'
%imdb.image_dir = '/work4/rbg/VOC2007/VOCdevkit/VOC2007/JPEGImages/'
%imdb.extension = '.jpg'
%imdb.image_ids = {'000001', ... }
%imdb.sizes = [numimages x 2]
%imdb.classes = {'aeroplane', ... }
%imdb.num_classes
%imdb.class_to_id
%imdb.class_ids
%imdb.eval_func = pointer to the function that evaluates detections
%imdb.roidb_func = pointer to the function that returns regions of interest
cache_file = ['./imdb/cache/kitti_' image_set];
try
  load(cache_file);
catch 
    imdb.name = ['kitti_' image_set];
    imdb.image_dir = image_dir;
    imdb.extension = 'png';
    images = [dir(fullfile(image_dir, '*.png'))];
    image_names = {images.name};
    imdb.image_ids = cellfun(@extractName,image_names, 'UniformOutput', false);

    imdb.classes = {'Car', 'Pedestrian'};
    imdb.num_classes = length(imdb.classes);
    imdb.class_to_id = containers.Map(imdb.classes, 1:imdb.num_classes);
    imdb.class_ids = 1:imdb.num_classes;
    imdb.eval_func = @imdb_eval_kitti;
    imdb.roidb_func = @roidb_from_kitti;
    imdb.image_at = @(i) ...
          sprintf('%s/%s.%s', imdb.image_dir, imdb.image_ids{i}, imdb.extension);

    for i = 1:length(imdb.image_ids)
        tic_toc_print('imdb (%s): %d/%d\n', imdb.name, i, length(imdb.image_ids));
        info = imfinfo(imdb.image_at(i));
        imdb.sizes(i, :) = [info.Height info.Width];
    end
    
    save(cache_file, 'imdb');
end
    
function name = extractName(filepath)
[~,name,~] = fileparts(filepath);

