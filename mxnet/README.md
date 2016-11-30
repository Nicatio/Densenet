# MXNet implementation for DenseNet

This repository contains the code for [Densely Connected Convolutional Networks](http://arxiv.org/abs/1608.06993). 

Based on [the examples in the mxnet](https://github.com/dmlc/mxnet/blob/master/example/image-classification) and [keras implementation for DenseNet](https://github.com/tdeboissiere/DeepLearningImplementations/tree/master/DenseNet), we implemented densenet for mxnet (in progress).

Currently, we are testing the performance.


## Usage

    python train_cifar10_densenet.py --batch-size 128 --lr 0.1 --gpus 0
