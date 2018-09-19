function prediction = LeNet(varargin)
%LENET Returns a simple LeNet-5 for digit classification
%   M = models.LeNet() returns a model inspired by:
%
%     LeCun et al., "Gradient-Based Learning Applied to Document
%     Recognition", Proceedings of the IEEE, 1998.
%
%   models.LeNet(..., 'option', value, ...) accepts the following options:
%
%   `input`:: default input
%     Specifies an input (images) layer for the network. If unspecified, a
%     new one is created.
%
%   `batchNorm`:: true
%     Whether to use batch normalization after each convolution.
%
%   Suggested SGD training options are also returned in the struct M.meta.

  % parse options
  opts.input = Input('name', 'images', 'gpu', true) ;  % default input layer
  opts.batchNorm = false ;  % whether to use batch-norm
  opts = vl_argparse(opts, varargin, 'nonrecursive') ;
  
  % build network
  images = opts.input ;
  
  x = vl_nnconv(images, 'size', [5, 5, 1, 20], 'weightScale', 0.01) ;
  if opts.batchNorm
    x = vl_nnbnorm(x) ;
  end
  x = vl_nnpool(x, 2, 'stride', 2) ;
  
  x = vl_nnconv(x, 'size', [5, 5, 20, 50], 'weightScale', 0.01) ;
  if opts.batchNorm
    x = vl_nnbnorm(x) ;
  end
  x = vl_nnpool(x, 2, 'stride', 2) ;
  
  x = vl_nnconv(x, 'size', [4, 4, 50, 500], 'weightScale', 0.01) ;
  if opts.batchNorm
    x = vl_nnbnorm(x) ;
  end
  x = vl_nnrelu(x) ;
  
  prediction = vl_nnconv(x, 'size', [1, 1, 500, 10], 'weightScale', 0.01) ;

  % default training options for this network
  defaults.numEpochs = 20 ;
  defaults.batchSize = 128 ;
  defaults.learningRate = 0.001 ;
  defaults.weightDecay = 0 ;
  defaults.imageSize = [28, 28, 1] ;
  prediction.meta = defaults ;
end
