function prediction = ResNet(varargin)
%ResNet Returns a ResNet-50/101/152/custom model for ImageNet
%   M = models.ResNet() returns the ResNet-50 model proposed in:
%
%     He et al., "Deep Residual Learning for Image Recognition", CVPR 2016.
%
%   models.ResNet(..., 'option', value, ...) accepts the following
%   options:
%
%   `variant`:: '50'
%     Model variant: 50, 101, 152 or 'custom'.
%
%   `pretrained`:: false
%     If true, returns a model pre-trained on ImageNet (using the
%     MatConvNet example code).
%
%   `input`:: default input
%     Specifies an input (images) layer for the network. If unspecified, a
%     new one is created.
%
%   `numClasses`:: 1000
%     Number of output classes.
%
%   `blocksPerSection`:: depends on variant
%     Vector with the number of blocks in each section. This is set
%     automatically depending on the variant, unless it is 'custom'. The
%     standard variants have only 4 sections (thus this is a 4-elements
%     vector), but different numbers of sections are supported.
%
%   `downsampleSection`:: [0, 1, 1, 1]
%     Whether to downsample spatially at each section (binary).
%
%   `channelsPerSection`:: [64, 128, 256, 512]
%     Base number of output channels in each section.
%
%   `channelsPerBlock`:: [1, 1, 4]
%     Modifying factor for number of channels per conv layer in a block.
%
%   `initialSize`:: [7, 7, 3, 64]
%     Filter size of the input convolution (4D tensor).
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
  % activation).
  opts.pretrained = false ;  % whether to fetch a pre-trained model
  opts.variant = '50';  % model variant: 50, 101, 152 or 'custom'
  opts.input = Input('name', 'images', 'gpu', true) ;  % default input layer
  opts.numClasses = 1000 ;  % number of predicted classes
  opts.blocksPerSection = [] ;  % vector with #blocks per section (depends on variant)
  opts.downsampleSection = [0, 1, 1, 1] ;  % whether to downsample in each section
  opts.channelsPerSection = [64, 128, 256, 512] ;  % #channels per section
  opts.channelsPerBlock = [1, 1, 4] ;  % modifying factor for #channels per conv in a block
  opts.initialSize = [7, 7, 3, 64] ;  % filter size of the input convolution (4D tensor)
  [opts, convBlockArgs] = vl_argparse(opts, varargin, 'nonrecursive') ;
  
  % set the number of blocks per section for standard ResNet variants
  if isnumeric(opts.variant)  % accept variant 50 instead of '50'
    opts.variant = int2str(opts.variant) ;
  end
  switch opts.variant
  case '18'
    opts.blocksPerSection = [2, 2, 2, 2] ;
    opts.channelsPerBlock(3:end) = [];  % only two convolutions per block
  case '34'
    opts.blocksPerSection = [3, 4, 6, 3] ;
    opts.channelsPerBlock(3:end) = [];  % only two convolutions per block
  case '50'
    opts.blocksPerSection = [3, 4, 6, 3] ;
  case '101'
    opts.blocksPerSection = [3, 4, 23, 3] ;
  case '152'
    opts.blocksPerSection = [3, 8, 36, 3] ;
  case 'custom'
    assert(~isempty(opts.blocksPerSection), ...
      'To build a custom ResNet, blocksPerSection must be specified.') ;
  otherwise
    error('Unknown variant.') ;
  end
  
  assert(any(numel(opts.channelsPerBlock) == [2, 3]), ...
    'Can only specify 2 or 3 channels per block.') ;
  
  
  % default training options for this network (returned as output.meta)
  meta.batchSize = 64 ;  % 256 with subbatches
  meta.imageSize = [224, 224, 3] ;
  meta.augmentation.crop = 224 / 256;
  meta.augmentation.location = true ;
  meta.augmentation.flip = true ;
  meta.augmentation.brightness = 0.1 ;
  meta.augmentation.aspect = [3/4, 4/3] ;
  meta.augmentation.scale = [0.4, 1.1] ;
  meta.weightDecay = 0.0001 ;
  meta.momentum = 0.9 ;
  
  % the default learning rate schedule
  if ~opts.pretrained
    meta.learningRate = [0.1 * ones(1,30), 0.01 * ones(1,30), 0.001 * ones(1,30)] ;
    meta.numEpochs = numel(meta.learningRate) ;
  else  % fine-tuning has lower LR
    meta.learningRate = 1e-5 ;
    meta.numEpochs = 20 ;
  end
  
  
  % return a pre-trained model
  if opts.pretrained
    if opts.numClasses ~= 1000
      warning('Model options are ignored when loading a pre-trained model.') ;
    end
    prediction = models.pretrained(['imagenet-resnet-' opts.variant '-dag']) ;
    
    % return prediction layer (not softmax)
    assert(isequal(prediction{1}.func, @vl_nnsoftmax)) ;
    prediction = prediction{1}.inputs{1} ;
    
    % replace input layer with the given one
    input = prediction.find('Input', 1) ;
    prediction.replace(input, opts.input) ;
    
    % in pre-trained networks, the last pooling is fixed to have size 7x7.
    % to make it compatible with any input size, replace it with mean().
    pool = prediction.inputs{1} ;
    assert(isequal(pool.func, @vl_nnpool));  % sanity check
    
    poolIn = pool.inputs{1} ;  % input to pool
    prediction.inputs{1} = mean(mean(poolIn, 1), 2) ;  % replace pooling
    
    prediction.meta = meta ;
    return
  end
  
  
  % get conv block generator with the given options (can be overriden)
  conv = models.ConvBlock('hasBias', false, 'pad', 'same', 'batchNorm', true, ...
    'preActivationBatchNorm', true, 'batchNormEpsilon', 1e-5, convBlockArgs{:}) ;
  
  % build network
  images = opts.input ;
  
  % add input section
  x = conv(images, 'size', opts.initialSize, 'stride', 2) ;
  x = vl_nnpool(x, 3, 'stride', 2, 'pad', 1)  ;
  channelsOut = opts.initialSize(4) ;
  
  for s = 1 : numel(opts.blocksPerSection)  % iterate sections
    for l = 1 : opts.blocksPerSection(s)  % iterate segments
      % output channels of each conv in this segment
      ch = opts.channelsPerSection(s) * opts.channelsPerBlock ;
      
      % downsample if first segment in section from section 2 onwards
      stride = 1 ;
      if l == 1 && opts.downsampleSection(s)
        stride = 2 ;
      end
      
      % this will be connected to the sum later
      sumInput = x ;
      
      if numel(ch) == 3
        % 3 convs for deeper variants: 1x1, 3x3, 1x1
        x = conv(x, 'size', [1, 1, channelsOut, ch(1)]) ;
        x = conv(x, 'size', [3, 3, ch(1), ch(2)], 'stride', stride) ;
        x = conv(x, 'size', [1, 1, ch(2), ch(3)], 'activation', 'none') ;
      else
        % 2 convs for shallower variants: 3x3, 3x3
        x = conv(x, 'size', [3, 3, channelsOut, ch(1)], 'stride', stride) ;
        x = conv(x, 'size', [3, 3, ch(1), ch(2)], 'activation', 'none') ;
      end

      % optional adapter layer, parallel to the above 3 convs
      if l == 1 && channelsOut ~= ch(end)
        sumInput = conv(sumInput, 'size', [1, 1, channelsOut, ch(end)], ...
          'stride', stride, 'activation', 'none') ;
      end

      % sum layer
      x = sumInput + x ;
      x = vl_nnrelu(x) ;
      
      channelsOut = ch(end);
    end
  end

  % average pooling
  x = mean(mean(x, 1), 2) ;
  
  % prediction layer
  prediction = conv(x, 'size', [1, 1, channelsOut, opts.numClasses], ...
    'batchNorm', false, 'activation', 'none', 'hasBias', true) ;

  prediction.meta = meta ;
  
end
