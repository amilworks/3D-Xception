# 3DXception
#### Author: Amil Khan
==============================

## Background
This repository consists of a PyTorch implementation of 3DXception, a convolutional neural network for video classification. The code enables users to train 3DXception for problems in video classification, but has not been tested for data outside videos such as volumetric data. This network is based on the successful Xception network by François Chollet. The paper is __Xception: Deep Learning with Depthwise Separable Convolutions__ and I highly recommend you check it out. For convenience, I will attach an abridged version of [Chollet's Xception Paper Abstract](https://arxiv.org/pdf/1610.02357):

> __Abstract:__ We present an interpretation of Inception modules in convolutional neural networks as being an intermediate step in-between regular convolution and the depthwise separable convolution operation (a depthwise convolution followed by a pointwise convolution). In this light, a depthwise separable convolution can be understood as an Inception module with a maximally large number of towers. This observation leads us to propose a novel deep convolutional neural network architecture inspired by Inception, where Inception modules have been replaced with depthwise separable convolutions.

## Prerequisites 

* Linux (Tested on Ubuntu 16.04 and 18.04)
* NVIDIA GPU
* CUDA
* Docker (Recommended)


## Getting Started

There are two routes you can take, but I highly suggest the Docker installation. Why? Because Docker is awesome. Okay, besides that, I have already packaged everything you need to get started with dataloading, visualizing, preprocessing, training, postprocessing, whatever you need, I probably got you.

__Pull Docker Container__
```
docker pull docker.pkg.github.com/amilworks/3d-xception/3d-xception:v0.1.0
```







