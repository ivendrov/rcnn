## R-CNN: *Regions with Convolutional Neural Network Features*

Created by Ross Girshick, Jeff Donahue, Trevor Darrell and Jitendra Malik at UC Berkeley EECS.

Acknowledgements: a huge thanks to Yangqing Jia for creating Caffe and the BVLC team, with a special shoutout to Evan Shelhamer, for maintaining Caffe and helping to merge the R-CNN fine-tuning code into Caffe.

---

Adapted by Ivan Vendrov at the University of Toronto to the [KITTI object detection dataset] (http://www.cvlibs.net/datasets/kitti/eval_object.php) for autonomous driving.

Summary of changes so far:

1. Removed rescaling all images to width 500 prior to region proposal generation (changes the number of regions from ~1000 to ~5000 per image) 
