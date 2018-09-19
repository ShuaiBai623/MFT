function prediction = NIN(varargin)
%NIN Returns a Network-in-Network model for CIFAR10
%   M = models.NIN() returns the model proposed in:
%
%     Lin et al, "Network in network", arXiv technical report 2013.
%
%   models.NIN(..., 'option', value, ...) accepts the following options:
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

% Copyright (C) 2018 Joao F. Henriques, Andrea Vedaldi.
% All rights reserved.
%
% This file is part of the VLFeat library and is made available under
% the terms of the BSD license (see the COPYING file).

  % parse options. unknown arguments will be passed to ConvBlock (e.g.
  % batchNorm).
  opts.input = Input('name', 'images', 'gpu', true) ;  % default input layer
  opts.numClasses = 10 ;  % number of predicted classes
  opts.batchNorm = true ;  % whether to use batch normalization
  [opts, convBlockArgs] = vl_argparse(opts, varargin, 'nonrecursive') ;
  
  % build network
  images = opts.input ;
  
  % first NIN block
  channels = [192 160 96] ;  % number of outputs channels per conv layer
  ker = [5 5] ;  % conv kernel
  poolKer = [3 3] ;  % pooling kernel
  poolMethod = 'max' ;  % pooling method
  pad = 2 ;  % input padding
  m1 = ninBlock(images, 3, channels, ker, pad, ...
    poolKer, poolMethod, convBlockArgs, opts.batchNorm, false) ;
  outChannels = channels(3) ;  % output channels of the NIN block
  
  % second NIN block
  channels = [192 192 192] ;
  ker = [5 5] ;
  poolKer = [3 3] ;
  poolMethod = 'avg' ;
  pad = 2 ;
  m2 = ninBlock(m1, outChannels, channels, ker, pad, ...
    poolKer, poolMethod, convBlockArgs, opts.batchNorm, false) ;
  outChannels = channels(3) ;
  
  % third NIN block
  channels = [192 192 opts.numClasses] ;
  ker = [3 3] ;
  poolKer = [7 7] ;
  poolMethod = 'avg' ;
  pad = 1 ;
  prediction = ninBlock(m2, outChannels, channels, ker, pad, ...
    poolKer, poolMethod, convBlockArgs, opts.batchNorm, true) ;
  
  
  % default training options for this network
  defaults.batchSize = 100 ;
  defaults.weightDecay = 0.0005 ;
  defaults.imageSize = [32, 32, 3] ;
  % the default learning rate schedule
  if ~opts.batchNorm
    defaults.learningRate = [0.002, 0.01, 0.02, 0.04 * ones(1,80), 0.004 * ones(1,10), 0.0004 * ones(1,10)] ;
    defaults.numEpochs = numel(defaults.learningRate) ;
  else
    defaults.learningRate = 0.01 ;
    defaults.numEpochs = 40 ;
  end
  prediction.meta = defaults ;
  
end

function block = ninBlock(in, inChannels, outChannels, ...
  ker, pad, poolKer, poolMethod, convBlockArgs, bnorm, final)

  % get conv block generator with the given options.
  % default activation is ReLU, with post-activation batch normalization.
  conv = models.ConvBlock('batchNorm', bnorm, convBlockArgs{:}) ;
  
  % if it's the final layer (prediction), the last conv block has no
  % activation or batch-norm
  finalArgs = {} ;
  if final
    finalArgs = {'activation', 'none', 'batchNorm', false} ;
  end
  
  % 3 conv blocks
  c1 = conv(in, 'size', [ker(1:2), inChannels, outChannels(1)], 'pad', pad) ;
  c2 = conv(c1, 'size', [1, 1, outChannels(1), outChannels(2)]) ;
  c3 = conv(c2, 'size', [1, 1, outChannels(2), outChannels(3)], finalArgs{:}) ;

  % pooling
  p1 = vl_nnpool(c3, poolKer, 'method', poolMethod, 'stride', 2) ;

  % dropout, skipped if it's the final layer (prediction)
  if ~final
    block = vl_nndropout(p1, 'rate', 0.5) ;
  else
    block = p1 ;
  end
end

