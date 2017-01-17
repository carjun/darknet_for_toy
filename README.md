![Darknet Logo](http://pjreddie.com/media/files/darknet-black-small.png)

#Darknet#
Darknet is an open source neural network framework written in C and CUDA. It is fast, easy to install, and supports CPU and GPU computation.

For more information see the [Darknet project website](http://pjreddie.com/darknet).

For questions or issues please use the [Google Group](https://groups.google.com/forum/#!forum/darknet).

#Darknet for toy#
This is a modified version to do the detection work on the 24-toy dataset.
In other words, it cannot be directy applied to other different datasets.
Except the function the original program has, I added three modes:
1. toymap. This mode will record precisions and recalls for different thresholds.
2. toyallframes. This mode will record the result images for all the frames.
3. toyboxes. This mode will record all the coordinates for all boxes in each image.
