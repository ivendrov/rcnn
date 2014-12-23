% clear and close everything
clear all; close all;

root_dir  = '~/kitti/object_detection';
train_dir = fullfile(root_dir,'training');
train_label_dir = fullfile(train_dir, 'label_2');
train_image_dir = fullfile(train_dir, 'image_2');
test_dir  = fullfile(root_dir,'/training/rcnn_label_2'); % location of your testing dir

addpath(fullfile(root_dir,'devkit/matlab'));


% read objects of first training image
train_objects = readLabels(train_label_dir,0);

% set RCNN 
rcnn_model_file = './data/rcnn_models/ilsvrc2013/rcnn_model.mat';

% Initialization only needs to happen once (so this time isn't counted
% when timing detection).
fprintf('Initializing R-CNN model (this might take a little while)\n');
use_gpu = 1;
thresh = -0.3;
rcnn_model = rcnn_load_model(rcnn_model_file, use_gpu); 
fprintf('done\n');

% loop over all images
for image = 0:10000;
    im = imread(sprintf('%s/%06d.png',train_image_dir,image));
    dets = rcnn_detect(im, rcnn_model, thresh);

    all_dets = [];
    for i = 1:length(dets)
      all_dets = cat(1, all_dets, ...
          [i * ones(size(dets{i}, 1), 1) dets{i}]);
    end

    [~, ord] = sort(all_dets(:,end), 'descend');
    test_objects = [];
    for i = 1:length(ord)
      score = all_dets(ord(i), end);
      classIndex = all_dets(ord(i),1);
      box = all_dets(ord(i),2:5);
      if score < thresh
        break;
      end
      
      className1 = rcnn_model.classes{all_dets(ord(i),1)};
      switch className1
          case 'car' 
              className = 'Car';
          case 'person'
              className = 'Pedestrian';
          otherwise
              className = 'DontCare';
      end
      
      test_objects(i).type  = className;
      test_objects(i).x1    = box(1);
      test_objects(i).y1    = box(2);
      test_objects(i).x2    = box(3);
      test_objects(i).y2    = box(4);
      test_objects(i).alpha = pi/2;
      test_objects(i).score = score;
      %cls = rcnn_model.classes{all_dets(ord(i), 1)};
      %showboxes(im, box);
      %title(sprintf('det #%d: %s score = %.3f', ...
      %    i, className1, score));
      %drawnow;
      %pause;
    end
    writeLabels(test_objects,test_dir,image);
end


% write object to file

disp('Test label file written!');
