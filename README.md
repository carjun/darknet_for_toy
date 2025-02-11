![Darknet Logo](http://pjreddie.com/media/files/darknet-black-small.png)

## Darknet

Darknet is an open source neural network framework written in C and CUDA. It is fast, easy to install, and supports CPU and GPU computation.

For more information see the [Darknet project website](http://pjreddie.com/darknet).

For questions or issues please use the [Google Group](https://groups.google.com/forum/#!forum/darknet).

### Darknet for toy

This is a modified version to do the detection work on the 24-toy dataset.

In other words, it cannot be directy applied to other different datasets.

A few steps to get started: 

- Add `$CUDA_PATH` to the `$PATH` variable. Verify if the path to CUDA in the `./Makefile` is correct. If you're running YOLO on Salk, the CUDA path in Makefile should be correct. 
- `cp /data/sbambach/yolo/trained_model_weights/yolo_15_toddlers.weights ./` 

## Running YOLO to detect objects:

`./darknet yolo getboxes $network_config_file $input_text_file $outout_text_file $trained_network_weights`

For example:

- `$network_config_file`: `cfg/yolo.cfg` (this is a text file that contains information about the architecture of the network)
- `$input_text_file`: `/data/sbambach/yolo/test_frame.txt` (this is a text file containing a list of images)
- `$output_text_file`: `/data/sbambach/yolo/test_output.txt` (this text file will be created by YOLO and contains the predicted boxes for each image)
- `$trained_network_weights`: `../trained_model_weights/yolo_15_toddlers.weights` (this is a binary files containing the weights of a trained network)`

`./darknet yolo getboxes cfg/yolo.cfg /data/sbambach/yolo/test_frame.txt /data/sbambach/yolo/test_output.txt ../trained_model_weights/yolo_15_toddlers.weights`


## Training YOLO:

For example:

- `$network_config_file`: `cfg/yolo.cfg` (this is a text file that contains information about the architecture of the network)
- `$input_text_file`: `/data/sbambach/yolo/dianzhi_scripts/training_data/training.txt` (this is a text file containing a list of images to use for training)
- `$new_trained_network_weights`: `/data/sbambach/yolo/trained_model_weights` (this is the directory where the binary files containing the newly trained network weights will be saved)
- `$pre_trained_network_weights`: `../trained_model_weights/yolo_15_toddlers.weights` (this parameter is OPTIONAL for training. if given, the training will proceed starting from the previously trained networks weights. if left out, the network will be trained from scratch, e.g. for a new dataset)

## Further train a previously trained model:

`./darknet yolo train cfg/yolo.cfg /data/sbambach/yolo/dianzhi_scripts/training_data/training.txt /data/sbambach/yolo/trained_model_weights ../trained_model_weights/yolo_15_toddlers.weights`

## Train model with pre-trained convolutional layers:

`./darknet yolo train cfg/yolo.train.cfg ../prepare_training_data/training_data_toyroom_parent/training.txt ../trained_model_weights ../trained_model_weights/extraction.conv.weights`

## Train a new model from scratch:

`./darknet yolo train cfg/yolo.cfg /data/sbambach/yolo/dianzhi_scripts/training_data/training.txt /data/sbambach/yolo/trained_model_weights`


`./darknet yolo train cfg/yolo.cfg /data/sbambach/yolo/prepare_training_data/training_data_toyroom_parent/training.txt /data/sbambach/yolo/trained_model_weights`

## Additional features: 

Except the modes the original Darknet has, I added three modes:

1. toymap. This mode will record precisions and recalls for different thresholds.

2. toyallframes. This mode will record the result images for all the frames.

3. toyboxes. This mode will record all the coordinates for all boxes in each image.

Their usages are the same as the recall, train, etc., modes in the original Darknet.


