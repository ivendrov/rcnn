% clear and close everything
close all;

root_dir  = '~/kitti/object_detection';
test_dir  = fullfile(root_dir,'/training/rcnn_label_2/val'); % location of your testing dir
addpath(fullfile(root_dir,'devkit/matlab'));
imdb = imdb_from_kitti('val');

% set RCNN 
rcnn_model_file = rcnn_create_model('./model-defs/kitti_finetune_deploy.prototxt', './finetuning/kitti/finetune_kitti_train_iter_41000.caffemodel');%'./cachedir/kitti_train/rcnn_model.mat';

% Initialization only needs to happen once (so this time isn't counted
% when timing detection).
fprintf('Initializing R-CNN model (this might take a little while)\n');
use_gpu = 1;
rcnn_model = rcnn_load_model(rcnn_model_file, use_gpu);
rcnn_model.classes = {'Car', 'Pedestrian', 'Cyclist'};
caffe('set_device', 1);
fprintf('done\n');

fprintf('Running test: \n')
aboxes = rcnn_test(rcnn_model, imdb);

for i = 1:length(imdb.image_ids)
    idx = 0;
    objects = [];
    for j = imdb.class_ids % number of classes
        detections = aboxes{j}{i};
        
        for k = 1:size(detections,1)
            idx = idx + 1;
            objects(idx).type = imdb.classes{j};
            objects(idx).score = detections(k,end);
            objects(idx).x1 = detections(k,1);
            objects(idx).y1 = detections(k,2);
            objects(idx).x2 = detections(k,3);
            objects(idx).y2 = detections(k,4);
            objects(idx).alpha = -10; 
        end
    end
           
    writeLabels(objects,test_dir,str2num(['int64(' imdb.image_ids{i} ')']));
end


% write object to file

disp('Test label file written!');
