# I3D models trained on Kinetics

## Overview

This repository contains trained models reported in the paper "[Quo Vadis,
Action Recognition? A New Model and the Kinetics
Dataset](https://arxiv.org/abs/1705.07750)" by Joao Carreira and Andrew
Zisserman. The paper was posted on arXiv in May 2017, and will be published as a
CVPR 2017 conference paper.

"Quo Vadis" introduced a new architecture for video classification, the Inflated
3D Convnet or I3D. Here we release Inception-v1 I3D models trained on the
[Kinetics dataset](www.deepmind.com/kinetics) training split.

In our paper, we reported state-of-the-art results on the UCF101 and HMDB51
datasets from fine-tuning these models. I3D models pre-trained on Kinetics also
placed first in the CVPR 2017 [Charades
challenge](http://vuchallenge.org/charades.html).

The repository also now includes a pre-trained checkpoint using rgb inputs and trained from scratch on Kinetics-600.

**NEW**: the video preprocessing we used has now been open-sourced by google. To set it up, check [these instructions in Google's MediaPipe repo](https://github.com/google/mediapipe/blob/master/mediapipe/docs/install.md).


Disclaimer: This is not an official Google product.

## Running the code

### Setup

First follow the instructions for [installing
Sonnet](https://github.com/deepmind/sonnet).

Then, clone this repository using

`$ git clone https://github.com/deepmind/kinetics-i3d`

### Sample code

Run the example code using

`$ python evaluate_sample.py`

With default flags, this builds the I3D two-stream model, loads pre-trained I3D
checkpoints into the TensorFlow session, and then passes an example video
through the model. The example video has been preprocessed, with RGB and Flow
NumPy arrays provided (see more details below).

The script outputs the norm of the logits tensor, as well as the top 20 Kinetics
classes predicted by the model with their probability and logit values. Using
the default flags, the output should resemble the following up to differences in
numerical precision:

```
Norm of logits: 138.468643

Top classes and probabilities
1.0 41.8137 playing cricket
1.49716e-09 21.494 hurling (sport)
3.84312e-10 20.1341 catching or throwing baseball
1.54923e-10 19.2256 catching or throwing softball
1.13602e-10 18.9154 hitting baseball
8.80112e-11 18.6601 playing tennis
2.44157e-11 17.3779 playing kickball
1.15319e-11 16.6278 playing squash or racquetball
6.13194e-12 15.9962 shooting goal (soccer)
4.39177e-12 15.6624 hammer throw
2.21341e-12 14.9772 golf putting
1.63072e-12 14.6717 throwing discus
1.54564e-12 14.6181 javelin throw
7.66915e-13 13.9173 pumping fist
5.19298e-13 13.5274 shot put
4.26817e-13 13.3313 celebrating
2.72057e-13 12.8809 applauding
1.8357e-13 12.4875 throwing ball
1.61348e-13 12.3585 dodgeball
1.13884e-13 12.0101 tap dancing
```

### Running the test

The test file can be run using

`$ python i3d_test.py`

This checks that the model can be built correctly and produces correct shapes.

## Further details

### Provided checkpoints

The default model has been pre-trained on ImageNet and then Kinetics; other
flags allow for loading a model pre-trained only on Kinetics and for selecting
only the RGB or Flow stream. The script `multi_evaluate.sh` shows how to run all
these combinations, generating the sample output in the `out/` directory.

The directory `data/checkpoints` contains the four checkpoints that were
trained. The ones just trained on Kinetics are initialized using the default
Sonnet / TensorFlow initializers, while the ones pre-trained on ImageNet are
initialized by bootstrapping the filters from a 2D Inception-v1 model into 3D,
as described in the paper. Importantly, the RGB and Flow streams are trained
separately, each with a softmax classification loss. During test time, we
combine the two streams by adding the logits with equal weighting, as shown in
the `evalute_sample.py` code.

We train using synchronous SGD using `tf.train.SyncReplicasOptimizer`. For each
of the RGB and Flow streams, we aggregate across 64 replicas with 4 backup
replicas. During training, we use 0.5 dropout and apply BatchNorm, with a
minibatch size of 6. The optimizer used is SGD with a momentum value of 0.9, and
we use 1e-7 weight decay. The RGB and Flow models are trained for 115k and 155k
steps respectively, with the following learning rate schedules.

RGB:

*   0 - 97k: 1e-1
*   97k - 108k: 1e-2
*   108k - 115k: 1e-3

Flow:

*   0 - 97k: 1e-1
*   97k - 104.5k: 1e-2
*   104.5k - 115k: 1e-3
*   115k - 140k: 1e-1
*   140k - 150k: 1e-2
*   150k - 155k: 1e-3

This is because the Flow models were determined to require more training after
an initial run of 115k steps.

The models are trained using the training split of Kinetics. On the Kinetics
test set, we obtain the following top-1 / top-5 accuracy:

Model          | ImageNet + Kinetics | Kinetics
-------------- | :-----------------: | -----------
RGB-I3D        | 71.1 / 89.3         | 68.4 / 88.0
Flow-I3D       | 63.4 / 84.9         | 61.5 / 83.4
Two-Stream I3D | 74.2 / 91.3         | 71.6 / 90.0

### Sample data and preprocessing

The release of the [DeepMind Kinetics dataset](www.deepmind.com/kinetics) only
included the YouTube IDs and the start and end times of the clips. For the
sample data here, we use a video from the UCF101 dataset, for which all the
videos are provided in full. The video used is `v_CricketShot_g04_c01.mp4` which
can be downloaded from the [UCF101
website](http://crcv.ucf.edu/data/UCF101.php).

Our preprocessing uses internal libraries, that have now been open-sourced [check Google's MediaPipe repo](https://github.com/google/mediapipe/blob/master/mediapipe/docs/install.md). It does the following: 
for both streams, we sample frames at 25 frames per second. For Kinetics, we
additionally clip the videos at the start and end times provided.

For RGB, the videos are resized preserving aspect ratio so that the smallest
dimension is 256 pixels, with bilinear interpolation. Pixel values are then
rescaled between -1 and 1. During training, we randomly select a 224x224 image
crop, while during test, we select the center 224x224 image crop from the video.
The provided `.npy` file thus has shape `(1, num_frames, 224, 224, 3)` for RGB,
corresponding to a batch size of 1.

For the Flow stream, after sampling the videos at 25 frames per second, we
convert the videos to grayscale. We apply a TV-L1 optical flow algorithm,
similar to [this code from
OpenCV](http://docs.opencv.org/3.1.0/d6/d39/classcv_1_1cuda_1_1OpticalFlowDual__TVL1.html).
Pixel values are truncated to the range [-20, 20], then rescaled between -1 and 1.
We only use the first two output dimensions, and apply the same cropping as
for RGB. The provided `.npy` file thus has shape `(1, num_frames, 224, 224, 2)`
for Flow, corresponding to a batch size of 1.

Here are gifs showing the provided `.npy` files. From the RGB data, we added 1
and then divided by 2 to rescale between 0 and 1. For the Flow data, we added a
third channel of all 0, then added 0.5 to the entire array, so that results are
also between 0 and 1.

![See
data/v_CricketShot_g04_c01_rgb.gif](data/v_CricketShot_g04_c01_rgb.gif "data/v_CricketShot_g04_c01_rgb.gif")

![See
data/v_CricketShot_g04_c01_flow.gif](data/v_CricketShot_g04_c01_flow.gif "data/v_CricketShot_g04_c01_flow.gif")

For additional details on preprocessing, check [this](https://github.com/google/mediapipe/blob/master/mediapipe/examples/desktop/media_sequence/kinetics_dataset.py), refer to our paper or contact
the authors.

### Acknowledgments

Brian Zhang, Joao Carreira, Viorica Patraucean, Diego de Las Casas, Chloe
Hillier, and Andrew Zisserman helped to prepare this initial release. We would
also like to thank the teams behind the [Kinetics
dataset](https://arxiv.org/abs/1705.06950) and the original [Inception
paper](https://arxiv.org/abs/1409.4842) on which this architecture and code is
based.

### Questions and contributions

To contribute to this repository, you will first need to sign the Google
Contributor License Agreement (CLA), provided in the CONTRIBUTING.md file. We
will then be able to accept any pull requests, though are not currently aiming
to expand to other trained models.

For any questions, you can contact the authors of the "Quo Vadis" paper, whose
emails are listed in the paper.
