function prediction = BasicCifarNet(varargin)
%BASICCIFARNET Returns a simple network for CIFAR10
%   M = models.BasicCifarNet() returns a model inspired by:
%
%     LeCun et al., "Gradient-Based Learning Applied to Document
%     Recognition", Proceedings of the IEEE, 1998.
%
%   models.BasicCifarNet(..., 'option', value, ...) accepts the following
%   options:
%
%   `input`:: default input
%     Specifies an input (images) layer for the network. If unspecified, a
%     new one is created.
%
%   `numClasses`:: 10
%     Number of output classes.
%
%   `batchNorm`:: true
%     Whether to use batch normalization.
%
%   Any other options will be passed to models.ConvBlock(), and can be used
%   to change the activation function, weight initialization, etc.
%
%   Suggested SGD training options are also returned in the struct M.meta.

  % parse options. unknown arguments will be passed to ConvBlock (e.g.
  % activation).
  opts.input = Input('name', 'images', 'gpu', true) ;  % default input layer
  opts.numClasses = 10 ;
  opts.batchNorm = true ;  % whether to use batch normalization
  [opts, convBlockArgs] = vl_argparse(opts, varargin, 'nonrecursive') ;
  
  % get conv block generator with the given options. default activation is
  % ReLU, with pre-activation batch normalization (can be overriden).
  conv = models.ConvBlock('batchNorm', opts.batchNorm, ...
    'preActivationBatchNorm', true, convBlockArgs{:}) ;
  
  % build network
  images = opts.input ;
  
  x = conv(images, 'size', [5, 5, 3, 32], 'pad', 2, 'weightScale', 0.01) ;
  x = vl_nnpool(x, 3, 'stride', 2, 'method', 'max', 'pad', 1) ;
  
  x = conv(x, 'size', [5, 5, 32, 32], 'pad', 2, 'weightScale', 0.05) ;
  x = vl_nnpool(x, 3, 'stride', 2, 'method', 'avg', 'pad', 1) ;
  
  x = conv(x, 'size', [5, 5, 32, 64], 'pad', 2, 'weightScale', 0.05) ;
  x = vl_nnpool(x, 3, 'stride', 2, 'method', 'avg', 'pad', 1) ;
  
  x = conv(x, 'size', [4, 4, 64, 64], 'weightScale', 0.05) ;
  
  prediction = conv(x, 'size', [1, 1, 64, opts.numClasses], 'weightScale', 0.05, ...
    'batchNorm', false, 'activation', 'none') ;
  
  
  % default training options for this network
  defaults.batchSize = 128 ;
  defaults.weightDecay = 0.0005 ;
  if ~opts.batchNorm
    defaults.learningRate = 0.01 ;
    defaults.numEpochs = 100 ;
  else
    defaults.learningRate = 0.1 ;
    defaults.numEpochs = 40 ;
  end
  defaults.imageSize = [32, 32, 3] ;
  prediction.meta = defaults ;
  
end
