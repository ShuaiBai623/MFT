function layer = vl_nnbrenorm_auto(varargin)
%VL_NNREBNORM_AUTO Additional options for vl_nnbrenorm (CNN batch normalisation)
%   Y = Layer.vl_nnbrenorm(X) applies batch renormalization to the input X,
%   creating all needed parameters automatically. See help vl_nnbrenorm for
%   more details.
%
%   This method overloads the vl_nnbrenorm function for Layer
%   objects, so that instead of executing vl_nnbrenorm, a new Layer object is
%   returned. Note also that, to maintain a uniform interface, during
%   network evaluation vl_nnbrenorm_wrapper is used instead of vl_nnbrenorm.
%
%   Y = Layer.vl_nnbrenorm(X, G, B, CLIPS) specifies the gains G, biases B, 
%   and clipping limits for `r` and `d`, CLIPS.  These may be other Layers, 
%   including Params, or constants.
%
%   Y = Layer.vl_nnbrenorm(..., 'moments', M) or Y = Layer.vl_nnbrenorm(..., M)
%   specifies the moments M. Note that the "derivative" for M returned by
%   vl_nnbrenorm is not a proper derivative, but an update for a moving
%   average. As such, only constants or Params with trainMethod = 'average'
%   are supported.
%
%   In addition to those defined by vl_nnbrenorm, the overloaded
%   VL_NNBRENORM(..., 'option', value, ...) accepts the following options:
%
%   `learningRate`:: [2 1 0.1]
%     Factor used to adjust the created Params' learning rate. Can specify
%     separate learning rates for G, B and M as a 3-elements vector.
%
%   `weightDecay`:: 0
%     Factor used to adjust the created Params' weight decay. Can specify
%     separate weight decays for G, B and M as a 3-elements vector.
%
%   `testMode`:: []
%     During training mode, the layer uses a combination of minibatch 
%     statistics and moments. In test mode, only the moments parameters
%     are used.


% Copyright (C) 2017 Samuel Albanie 
% (based on the vl_nnbnorm function by Joao F. Henriques)
% All rights reserved.

  % parse options. note the defaults for brenorm's Params are set here.
  opts = struct('learningRate', [2 1 0.1], 'weightDecay', 0, ...
                 'moments', [], 'testMode', []) ;
  [opts, posArgs, brenormOpts] = vl_argparsepos(opts, varargin) ;
  
  if isscalar(opts.learningRate)
    opts.learningRate = opts.learningRate([1 1 1]) ;
  end

  if isscalar(opts.weightDecay)
    opts.weightDecay = opts.weightDecay([1 1 1]) ;
  end
  
  % create any unspecified parameters (scale, bias and moments).
  assert(numel(posArgs) >= 1 && numel(posArgs) <= 4, ...
    'Must specify 1 to 4 inputs to VL_NNBRENORM, plus any name-value pairs.') ;
  
  if numel(posArgs) < 3
    % create scale param. will be initialized with proper number of
    % channels on first run by the wrapper.
    g = Param('value', single(1), ...
                      'learningRate', opts.learningRate(1), ...
                      'weightDecay', opts.weightDecay(1)) ;
  end
  
  if numel(posArgs) < 4
    % create bias param
    b = Param('value', single(0), ...
                      'learningRate', opts.learningRate(2), ...
                      'weightDecay', opts.weightDecay(2)) ;
  end

  
  if ~isempty(opts.moments)
    moments = opts.moments ;
  else
    % 'moments' name-value pair not specified.
    % check if the moments were passed in as the 4th argument (alternative
    % syntax)
    if numel(posArgs) > 4
      moments = posArgs{5} ;
      posArgs(5) = [] ;  % remove from list
    else
      % create moments param. note the training method is 'average'.
      moments = Param('value', single(0), ...
                      'learningRate', opts.learningRate(3), ...
                      'weightDecay', opts.weightDecay(3), ...
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
  
  x = posArgs{1} ;
  clips = posArgs{2} ;
  layer = Layer(@vl_nnbrenorm_wrapper, x, g, b, moments, clips, ...
                                                   testMode, brenormOpts{:}) ;
  layer.numInputDer = 4 ;
end
