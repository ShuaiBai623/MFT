function prediction = MaxoutNIN(varargin)
%MAXOUTNIN Returns a Maxout Network-in-Network for CIFAR10
%   M = models.MaxoutNIN() returns the model proposed in:
%
%     Chang and Chen, "Batch-normalized maxout network in network", arXiv
%     technical report, 2015.
%
%   This network is notable for achieving low error on CIFAR10 (~8%) with
%   a relatively short training time (compared to e.g. ResNet variants).
%
%   models.MaxoutNIN(..., 'option', value, ...) accepts the following options:
%
%   `input`:: default input
%     Specifies an input (images) layer for the network. If unspecified, a
%     new one is created.
%
%   `numClasses`:: 10
%     Number of output classes.
%
%   Any other options will be passed to models.ConvBlock(), and can be used
%   to change the activation function, weight initialization, etc.
%
%   Suggested SGD training options are also returned in the struct M.meta.

% Copyright (C) 2018 Samuel Albanie, Jia-Ren Chang, Joao F. Henriques.
% All rights reserved.
%
% This file is part of the VLFeat library and is made available under
% the terms of the BSD license (see the COPYING file).

  % parse options
  opts.input = Input('name', 'images', 'gpu', true) ;  % default input layer
  opts.numClasses = 10 ;
  [opts, convBlockArgs] = vl_argparse(opts, varargin, 'nonrecursive') ;
  
  % build network
  images = opts.input ;
  
  % first NIN block
  units = [192 160 96] ;  % maxout units per conv layer
  pieces = [1 5 5] ;  % maxout pieces per conv layer
  ker = [5 5] ;  % conv kernel
  poolKer = [3 3] ;  % pooling kernel
  pad = 2 ;  % input padding
  m1 = ninMaxoutBlock(images, 3, units, pieces, ...
    ker, pad, poolKer, convBlockArgs, false) ;
  outChannels = units(3) ;  % output channels of the NIN block
  
  % second NIN block
  units = [192 192 192] ;
  pieces = [1 5 5] ;
  ker = [5 5] ;
  poolKer = [3 3] ;
  pad = 2 ;
  m2 = ninMaxoutBlock(m1, outChannels, units, pieces, ...
    ker, pad, poolKer, convBlockArgs, false) ;
  outChannels = units(3) ;
  
  % third NIN block
  units = [192 192 opts.numClasses] ;
  pieces = [1 5 5] ;
  ker = [3 3] ;
  poolKer = [8 8] ;
  pad = 1 ;
  prediction = ninMaxoutBlock(m2, outChannels, units, pieces, ...
    ker, pad, poolKer, convBlockArgs, true) ;
  
  
  % default training options for this network
  defaults.numEpochs = 200 ;
  defaults.batchSize = 100 ;
  defaults.weightDecay = 0.0005 ;
  defaults.imageSize = [32, 32, 3] ;
  % the default learning rate schedule, changing every 50 epochs
  ep50 = ones(1, 50) ;
  defaults.learningRate = [0.5 * ep50, 0.05 * ep50, 0.005 * ep50, 0.0005 * ep50] ;
  prediction.meta = defaults ;
  
end

function block = ninMaxoutBlock(in, inChannels, units, pieces, ...
  ker, pad, poolKer, convBlockArgs, final)
  % helper function to create each of the 3 main blocks of the model

  % get conv block generator with the given options. by default there's no
  % activation (due to the presence of maxout), and batch normalization.
  conv = models.ConvBlock('activation', 'none', 'batchNorm', true, convBlockArgs{:}) ;
  
  % first conv block
  sz = [ker(1:2), inChannels, units(1) * pieces(1)] ;
  c1 = conv(in, 'size', sz, 'pad', pad) ;
  
  % second conv block
  sz = [1, 1, units(1) * pieces(1), units(2) * pieces(2)] ;
  c2 = conv(c1, 'size', sz) ;
  
  % first maxout block
  m1 = vl_nnmaxout(c2, pieces(2)) ;
  
  % third conv block
  sz = [1, 1, units(2), units(3) * pieces(3)] ;
  c3 = conv(m1, 'size', sz) ;

  % second maxout block
  m2 = vl_nnmaxout(c3, pieces(3)) ;

  % pooling
  p1 = vl_nnpool(m2, poolKer, 'method', 'avg', 'stride', 2, 'pad', [0 1 0 1]) ;

  % dropout, skipped if it's the final layer (prediction)
  if ~final
    block = vl_nndropout(p1, 'rate', 0.5) ;
  else
    block = p1 ;
  end
end

