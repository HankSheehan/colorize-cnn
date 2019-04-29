# colorize-cnn
A convolutional neural network to colorize grayscale images.

## Abstract
This project is to explore the possibilites of colorizing grayscale images using convolutional neural network. There should be an input of a grayscale image of any dimension and an output of an RGB image of the same dimension, representing the grayscale image in color.

## Overview
The neural network is created with Python using the [Keras](https://keras.io/) framework on top of [TensorFlow](https://www.tensorflow.org/).

## Dataset
The dataset used is the [CIFAR-10](https://www.cs.toronto.edu/~kriz/cifar.html) dataset.

## Setup

## Hardware
This model will trained on my 13-inch Macbook with no GPU, so let's see how it goes ¯\\\_(ツ)\_/¯.

The specs of the machine used are the following:
- **CPU**: 2.3 GHz Intel Core i5
- **Memory**: 16 GB 2133 MHz LPDDR3
- **Graphics**: Intel Iris Plus Graphics 640 1536 MB
- **Model**: MacBook Pro (13-inch, 2017, Two Thunderbolt 3 ports)

## The Experiment
### Flatten -> Dense
I began with a single dense layer. In order to have the dense layer work with a 32 by 32 by 1 image (32x32, 1 grayscale channel), I had to flatten the image to a 1024 length array. This dense layer would output a 3072 length array, 3 values for each pixel. This was a good starting place since it got me used to normalizing the images and training a network in this way. The results look good for a neural network that is essentially just mapping a single channel to three channels without any context.

![Figure 1](README-assets/flatten-dense1.png)

![Figure 2](README-assets/flatten-dense2.png)

![Figure 3](README-assets/flatten-dense3.png)




## Results
