## mcnExtraLayers

This repo contains a collection of common MatConvNet functions and DagNN layers which are shared across a number of classification and object detection frameworks.

### Layers:

* `vl_nnmax` - element-wise maximum across tensors
* `vl_nnsum` - element-wise sum across tensors
* `vl_nninterp` - a wrapper for bilinear interpolation
* `vl_nnslice` - slicing along a given dimension
* `vl_nnspatialsoftmax` - spatial application of the softmax operator
* `vl_nnreshape` -  tensor reshaping
* `vl_nnchannelshuffle` -  channel shuffling (introduced in [ShuffleNet](https://arxiv.org/abs/1707.01083))
* `vl_nnflatten` - flatten along a given dimension
* `vl_nnglobalpool` - global pooling
* `vl_nnsoftmaxt` - softmax along a given dimension
* `vl_nncrop_wrapper` - autonn function wrapper for `vl_nncrop.m`
* `vl_nnaxpy` - vector op `y <- a*x + y` (BLAS Level One style naming convention)
* `vl_nngnorm` - group normalization (an alternative to batch norm)
* `vl_nnhuberloss` - computation of the Huber (L1-smooth) loss
* `vl_nneuclidenaloss` - computation of the Euclidean (L2-smooth) loss
* `vl_nntukeyloss` - computation of Tukey's Biweight (robust) loss
* `vl_nnsoftmaxceloss` - soft-target cross entropy loss (operates on logits)
* `vl_nncaffepool` - "caffe-style" pooling (applies padding before pooling kernel)
* `vl_nnl2norm` - l2 feature normalisation

### Utilities

The module also contains some additional utilities which may be useful during network training:

* [findBestCheckpoint](https://github.com/albanie/mcnExtraLayers/blob/master/utils/findBestCheckpoint.m) - function to rank and prune network checkpoints saved during training (useful for saving space automatically at the end of a training run
* [checkLearningParams](https://github.com/albanie/mcnExtraLayers/blob/master/utils/checkLearningParams.m) - compare mcn network against a caffe prototxt

### Install

The module is easiest to install with the `vl_contrib` package manager:

```
vl_contrib('install', 'mcnExtraLayers') ;
vl_contrib('setup', 'mcnExtraLayers') ;
vl_contrib('test', 'mcnExtraLayers') ; % optional
```
