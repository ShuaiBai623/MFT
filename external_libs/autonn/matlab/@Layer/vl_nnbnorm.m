function layer = vl_nnbnorm(varargin)
%VL_NNBNORM Additional options for vl_nnbnorm (CNN batch normalisation)
%   Y = Layer.vl_nnbnorm(X) applies batch normalization to the input X,
%   creating all needed parameters automatically. See help vl_nnbnorm for
%   more details.
%
%   This method overloads MatConvNet's vl_nnbnorm function for Layer
%   objects, so that instead of executing vl_nnbnorm, a new Layer object is
%   returned. Note also that, to maintain a uniform interface, during
%   network evaluation vl_nnbnorm_wrapper is used instead of vl_nnbnorm.
%
%   Y = Layer.vl_nnbnorm(X, G, B) specifies the gains G and biases B. These
%   may be other Layers, including Params, or constants.
%
%   Y = Layer.vl_nnbnorm(..., 'moments', M) or Y = Layer.vl_nnbnorm(..., M)
%   specifies the moments M. Note that the "derivative" for M returned by
%   vl_nnbnorm is not a proper derivative, but an update for a moving
%   average. As such, only constants or Params with trainMethod = 'average'
%   are supported.
%
%   In addition to those defined by MatConvNet's vl_nnbnorm, the overloaded
%   VL_NNBNORM(..., 'option', value, ...) accepts the following options:
%
%   `learningRate`:: [2 1 0.1]
%     Factor used to adjust the created Params' learning rate. Can specify
%     separate learning rates for G, B and M as a 3-elements vector.
%
%   `weightDecay`:: 0
%     Factor used to adjust the created Params' weight decay. Can specify
%     separate weight decays for G and B as a 2-elements vector.
%
%   `testMode`:: []
%     By default, the layer uses batch statistics when evaluating the
%     network in training mode, and uses the moments parameters M when in
%     test mode.
%     If `testMode` is true, the layer will always run in test mode, and if
%     false, it will always run in training mode.

% Copyright (C) 2016 Joao F. Henriques.
% All rights reserved.
%
% This file is part of the VLFeat library and is made available under
% the terms of the BSD license (see the COPYING file).

  % parse options. note the defaults for bnorm's Params are set here.
  opts = struct('learningRate', [2 1 0.1], 'weightDecay', [0 0], ...
    'moments', [], 'testMode', []) ;
  [opts, posArgs, bnormOpts] = vl_argparsepos(opts, varargin, ...
    'flags', {'CuDNN', 'NoCuDNN'}) ;
  
  if isscalar(opts.learningRate)
    opts.learningRate = opts.learningRate([1 1 1]) ;
  end
  if isscalar(opts.weightDecay)
    opts.weightDecay = opts.weightDecay([1 1 1]) ;
  end
  
  % create any unspecified parameters (scale, bias and moments).
  assert(numel(posArgs) >= 1 && numel(posArgs) <= 4, ...
    'Must specify between 1 and 4 inputs to VL_NNBNORM, plus any name-value pairs.') ;
  
  if numel(posArgs) < 2
    % create scale param. will be initialized with proper number of
    % channels on first run by the wrapper.
    posArgs{2} = Param('value', single(1), ...
                      'learningRate', opts.learningRate(1), ...
                      'weightDecay', opts.weightDecay(1)) ;
  end
  
  if numel(posArgs) < 3
    % create bias param
    posArgs{3} = Param('value', single(0), ...
                      'learningRate', opts.learningRate(2), ...
                      'weightDecay', opts.weightDecay(2)) ;
  end
  
  if ~isempty(opts.moments)
    moments = opts.moments ;
  else
    % 'moments' name-value pair not specified.
    % check if the moments were passed in as the 4th argument (alternative
    % syntax)
    if numel(posArgs) > 3
      moments = posArgs{4} ;
      posArgs(4) = [] ;  % remove from list
    else
      % create moments param. note the training method is 'average'.
      moments = Param('value', single(0), ...
                      'learningRate', opts.learningRate(3), ...
                      'weightDecay', 0, ...
                      'trainMethod', 'average') ;
    end
  end
  
  assert(isnumeric(moments) || (isa(moments, 'Param') && ...
    strcmp(moments.trainMethod, 'average')), ...
    'Moments must be constant or a Param with trainMethod = ''average''.') ;
  
  % create Input('testMode') to know when in test mode
  testMode = opts.testMode ;  % might override with boolean constant
  if isempty(testMode)
    testMode = Input('testMode') ;
  end
  
  % create layer.
  % in normal mode, pass in moments so its derivatives are expected.
  layer = Layer(@vl_nnbnorm_wrapper, posArgs{:}, moments, testMode, bnormOpts{:}) ;
  
  layer.numInputDer = 4 ;
  
end

