function rcnn_exp_cache_features_ilsvrc13(chunk)

% -------------------- CONFIG --------------------
net_file     = './data/caffe_nets/ilsvrc_2012_train_iter_310k';
cache_name   = 'v1_caffe_imagenet_train_iter_310k';
crop_mode    = 'warp';
crop_padding = 16;

% change to point to your VOCdevkit install
devkit = './datasets/ILSVRC13';
% ------------------------------------------------

imdb_val1   = imdb_from_ilsvrc13(devkit, 'val1');
imdb_val2   = imdb_from_ilsvrc13(devkit, 'val2');
%imdb_test  = imdb_from_voc(VOCdevkit, 'test', '2007');

switch chunk
  case 'val1_1'
    end_at = ceil(length(imdb_val1.image_ids)/2);
    rcnn_cache_pool5_features(imdb_val1, ...
        'start', 1, 'end', end_at, ...
        'crop_mode', crop_mode, ...
        'crop_padding', crop_padding, ...
        'net_file', net_file, ...
        'cache_name', cache_name);
  case 'val1_2'
    start_at = ceil(length(imdb_val1.image_ids)/2)+1;
    rcnn_cache_pool5_features(imdb_val1, ...
        'start', start_at, ...
        'crop_mode', crop_mode, ...
        'crop_padding', crop_padding, ...
        'net_file', net_file, ...
        'cache_name', cache_name);
  case 'val2_1'
    end_at = ceil(length(imdb_val2.image_ids)/2);
    rcnn_cache_pool5_features(imdb_val2, ...
        'start', 1, 'end', end_at, ...
        'crop_mode', crop_mode, ...
        'crop_padding', crop_padding, ...
        'net_file', net_file, ...
        'cache_name', cache_name);
  case 'val2_2'
    start_at = ceil(length(imdb_val2.image_ids)/2)+1;
    rcnn_cache_pool5_features(imdb_val2, ...
        'start', start_at, ...
        'crop_mode', crop_mode, ...
        'crop_padding', crop_padding, ...
        'net_file', net_file, ...
        'cache_name', cache_name);

  case 'train'
    for i = 1:200
      imdb_train = imdb_from_ilsvrc13(devkit, ['train_pos_' num2str(i)]);
      rcnn_cache_pool5_features(imdb_train, ...
          'crop_mode', crop_mode, ...
          'crop_padding', crop_padding, ...
          'net_file', net_file, ...
          'cache_name', cache_name);
    end
end