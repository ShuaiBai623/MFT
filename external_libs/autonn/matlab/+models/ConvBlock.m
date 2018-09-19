function output = ConvBlock(varargin)
%CONVBLOCK Conv layer generator, with activation and optional batch-norm
%   F = models.ConvBlock() returns a function that generates convolutional
%   blocks. By default, they consist of a convolution followed by a ReLU
%   layer, but more options are available.
%
%   For example:
%
%    conv = models.ConvBlock() ;  % default is a 3x3 filter kernel
%    images = Input() ;
%    x = conv(images, 'channels', [1, 8]) ;  % no need to specify ReLU
%    x = conv(x, 'channels', [8, 10]) ;      % also kernel size is reused
%
%   is equivalent to:
%
%    images = Input() ;
%    x = vl_nnconv(images, 'size', [3 3 1 10]) ; % 3x3, 1 channel in, 8 out
%    x = vl_nnrelu(x) ;                          % first ReLU
%    x = vl_nnconv(x, 'size', [3 3 8 10]) ;    % 3x3, 8 channels in, 10 out
%    x = vl_nnrelu(x) ;                          % second ReLU
%
%   ConvBlock makes it easy to re-use the same options for many convolution
%   layers at the same time, with less repetition. For example, this
%   one-line change would add batch-norm layers and change the activation:
%
%    conv = models.ConvBlock('batchNorm', true, 'activation', 'leakyrelu')
%
%   The options can be overriden later when calling the generator function
%   (for example, setting ('activation', 'none') only for the final layer).
%   See the 'autonn/matlab/+models' folder for more examples.
%
%   All the options accepted by vl_nnconv are supported, namely: size,
%   stride, pad, dilate, weightScale, hasBias, learningRate, weightDecay,
%   transpose. See 'help Layer.vl_nnconv' for more details.
%
%   In addition to vl_nnconv's options, the following are accepted:
%
%   `activation`:: 'ReLU'
%     Activation function to use (case-insensitive). One of: 'ReLU',
%     'LeakyReLU', 'Sigmoid', 'None'.
%
%   `leak`:: 0.1
%     Leak parameter, only used if the activation is 'LeakyReLU'.
%
%   `batchNorm`:: false
%     Whether to create a batch normalization layer. Disabled by default.
%
%   `preActivationBatchNorm`:: false
%     Specifies whether batch-norm is placed before the activation, or
%     after (the default).
%
%   `batchNormScaleBias`:: true
%     Specifies whether batch-norm is followed by learnable scale and bias.
%
%   `batchNormEpsilon`:: 1e-5
%     Epsilon parameter of batch-norm (to prevent division-by-zero).
%
%   `batchNormCuDNN`:: false
%     Whether batch-norm uses CuDNN (does not seem to affect speed).
%
%   `kernel`:: [3, 3]
%     Specifies the kernel size. Only used if 'channels' is defined.
%
%   `channels`:: undefined
%     Specifies the number of input and output channels, as a 2-elements
%     vector. Note that setting 'kernel' and 'channels' is equivalent to
%     specifying the 'size' option (parameters tensor size), but if both
%     are present, 'size' takes precedence.
%
%   OUT = models.ConvBlock(IN, ...) immediately creates a conv block given
%   an input layer IN, instead of returning a generator function.

% Copyright (C) 2018 Joao F. Henriques.
% All rights reserved.
%
% This file is part of the VLFeat library and is made available under
% the terms of the BSD license (see the COPYING file).

  if nargin > 0 && isa(varargin{1}, 'Layer')
    % create conv block immediately, with given layer as input
    output = createConvBlock(varargin{:}) ;
  else
    % return generator
    args = varargin ;  % generic arguments
    output = @(inputLayer, varargin) createConvBlock(inputLayer, args{:}, varargin{:}) ;
  end
end

function out = createConvBlock(in, varargin)
  % parse options
  opts.kernel = [3, 3] ;
  opts.channels = [] ;
  opts.size = [] ;
  opts.batchNorm = false ;
  opts.batchNormScaleBias = true ;
  opts.batchNormEpsilon = 1e-5 ;
  opts.batchNormCuDNN = false ;
  opts.preActivationBatchNorm = false ;
  opts.activation = 'relu' ;
  opts.leak = 0.1 ;  % for leaky ReLU only
  [opts, convArgs] = vl_argparse(opts, varargin) ;
  
  % make sure only valid convolution options remain
  allowedConvArgs = {'stride', 'dilate', 'pad', 'same', 'verbose', 'cuDNN', 'noCuDNN', ...
    'cuDNNWorkspaceLimit', 'noDerData', 'noDerFilters', 'noDerBiases', ...
    'size', 'weightScale', 'hasBias', 'learningRate', 'weightDecay', 'transpose'} ;
  for i = 1:numel(convArgs)
    if ischar(convArgs{i})
      assert(any(strcmpi(convArgs{i}, allowedConvArgs)), ...
        ['Unknown argument given for convolution: ' convArgs{i} '.']) ;
    end
  end
  
  % specified kernel and channels instead of size
  if isempty(opts.size)
    % handle scalar kernel size (kernel is square)
    if isscalar(opts.kernel)
      opts.kernel = opts.kernel * [1, 1] ;
    end

    assert(numel(opts.kernel) == 2, 'Must specify kernel size as a 1 or 2-elements vector.') ;
    assert(numel(opts.channels) == 2, ['Must specify the number of input and ' ...
      'output channels for convolution, as a 2-elements vector in option `channels`.']) ;
    
    opts.size = [opts.kernel(:); opts.channels(:)]' ;
  end
  
  % create conv layer
  out = vl_nnconv(in, 'size', opts.size, convArgs{:}) ;
  
  % prepare batch-norm arguments list
  if opts.batchNorm
    bnormArgs = {} ;
    if ~opts.batchNormScaleBias  % fixed scale and bias (constants instead of Params)
      bnormArgs(end+1:end+2) = {1, 0} ;
    end
    if opts.batchNormCuDNN
      bnormArgs{end+1} = 'CuDNN' ;
    else
      bnormArgs{end+1} = 'NoCuDNN' ;
    end
  end
  
  % create pre-activation batch norm
  if opts.batchNorm && opts.preActivationBatchNorm
    out = vl_nnbnorm(out, bnormArgs{:}, 'epsilon', opts.batchNormEpsilon) ;
  end
  
  % create activation layer
  out = createActivation(out, opts);
  
  % create post-activation batch norm
  if opts.batchNorm && ~opts.preActivationBatchNorm
    out = vl_nnbnorm(out, bnormArgs{:}) ;
  end
end

function out = createActivation(in, opts)
  % create activation layer
  if isa(opts.activation, 'function_handle')
    out = opts.activation(in) ;  % custom function handle
    
  elseif isempty(opts.activation)
    out = in ;  % no activation
  else
    % standard activations
    switch lower(opts.activation)
    case 'relu'
      out = vl_nnrelu(in) ;
    case 'leakyrelu'
      out = vl_nnrelu(in, 'leak', opts.leak) ;
    case 'sigmoid'
      out = vl_nnsigmoid(in) ;
    case 'none'
      out = in ;
    otherwise
      error('Unknown activation type.') ;
    end
  end
end
