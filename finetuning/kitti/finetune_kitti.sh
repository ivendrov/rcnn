GLOG_logtostderr=1 /pkgs/caffe-master/build/tools/caffe.bin train -solver "kitti_finetune_solver.prototxt" -weights "../../data/caffe_nets/ilsvrc_2012_train_iter_310k" 2>&1 | tee log.txt
