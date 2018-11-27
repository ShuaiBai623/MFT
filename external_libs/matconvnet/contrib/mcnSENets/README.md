Squeeze-and-Excitation Networks
---

This directory contains code to evaluate the classification models released by
the authors of the [paper](https://arxiv.org/abs/1709.01507):

```
Squeeze-and-Excitation Networks, 
Jie Hu, Li Shen, Gang Sun, arxiv 2017
```

This code is based on the [original](https://github.com/hujie-frank/SENet) 
implementation (which uses caffe).

### Pretrained Models

Each of the Squeeze-and-Excitation networks released by the authors has been imported into [MatConvNet](https://github.com/vlfeat/matconvnet) and can be downloaded here:

[SE Networks](http://www.robots.ox.ac.uk/~albanie/mcn-models.html#se-models)

The `run_se_benchmarks.m` script will evaluate each of these models on the ImageNet validation set. It will download the models automatically if you have not already done so (note that these evaluations require a copy of the imagenet data).  The results of the evaluations are given below - note there are minor differences to the original scores (listed under `official`) due to variations in preprocessing (full details of the evaluation can be found [here](http://www.robots.ox.ac.uk/~albanie/models.html#se-models)):


| model	  | top-1 error (offical)	| top-5 error (official) |
|---------------------------|-------------------------|------------------------|
| SE-ResNet-50-mcn	        | 22.30 (22.37) | 6.30  (6.36) |
| SE-ResNet-101-mcn	        | 21.59 (21.75) | 5.81  (5.72) |
| SE-ResNet-152-mcn	        | 21.38 (21.34) | 5.60  (5.54) |
| SE-BN-Inception-mcn       | 24.16 (23.62) | 7.35  (7.04) |
| SE-ResNeXt-50-32x4d-mcn   | 21.01 (20.97) | 5.58  (5.54) |
| SE-ResNeXt-101-32x4d-mcn  | 19.73 (19.81) | 4.98  (4.96) |
| SENet-mcn	                | 18.67 (18.68) | 4.50  (4.47) |

There may be some difference in how the Inception network should be preprocessed relative to the others (this model exhibits a noticeable degradation). To give some idea of the relative computational burdens of each model, esimates are provided below:


| model | input size | param memory | feature memory | flops |
|-------|------------|--------------|----------------|-------|
| [SE-ResNet-50](reports/SE-ResNet-50.md) | 224 x 224 | 107 MB | 103 MB | 4 GFLOPs                |
| [SE-ResNet-101](reports/SE-ResNet-101.md) | 224 x 224 | 189 MB | 155 MB | 8 GFLOPs              |
| [SE-ResNet-152](reports/SE-ResNet-152.md) | 224 x 224 | 255 MB | 220 MB | 11 GFLOPs             |
| [SE-BN-Inception](reports/SE-BN-Inception.md) | 224 x 224 | 46 MB | 43 MB | 2 GFLOPs            |
| [SE-ResNeXt-50-32x4d](reports/SE-ResNeXt-50-32x4d.md) | 224 x 224 | 105 MB | 132 MB | 4 GFLOPs  |
| [SE-ResNeXt-101-32x4d](reports/SE-ResNeXt-101-32x4d.md) | 224 x 224 | 187 MB | 197 MB | 8 GFLOPs|
| [SENet](reports/SENet.md) | 224 x 224 | 440 MB | 347 MB | 21 GFLOPs                             |


Each estimate corresponds to computing a single element batch. This table was generated
with [convnet-burden](https://github.com/albanie/convnet-burden) - the repo has a list of the assumptions used produce estimations. Clicking on the model name should give a more detailed breakdown.

### Dependencies

This code uses the following two modules: 

* [autonn](https://github.com/vlfeat/autonn) - a wrapper for MatConvNet
* [mcnExtraLayers](https://github.com/albanie/mcnExtraLayers) - some useful additional layers

Both of these can be setup directly with `vl_contrib` (i.e. run `vl_contrib install <module-name>` then `vl_contrib setup <module-name>`).

### Notes



### Installation

The easiest way to use this module is to install it with the `vl_contrib` 
package manager:

```
vl_contrib('install', 'mcnSENets') ;
vl_contrib('setup', 'mcnSENets') ;
vl_contrib('test', 'mcnSENets') ; % optional
```


The ordering of the imagenet labels differs from the standard ordering commonly found in caffe, pytorch etc.  These are remapped automically in the evaluation code.  The mapping between the synsets indices can be found [here](misc/label_map.txt).
