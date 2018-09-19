classdef Param < Layer
%PARAM Defines a learnable network parameter
%   Defines a network parameter, such as a convolution's filter bank.
%
%   Note that some functions (e.g. vl_nnconv, vl_nnbnorm) can create and
%   initialize Param objects automatically. See help Layer.vl_nnconv,
%   Layer.vl_nnbnorm for more details.
%
%   Param('option', value, ...) accepts the following options:
%
%   `value`:: []
%     Initial parameter value. This must always be specified.
%
%   `name`:: []
%     Specifies the layer name, which is optional. Names can be filled in
%     automatically with Layer.workspaceNames() or obj.sequentialNames().
%
%   `gpu`:: true
%     Automatically moves the parameter to the GPU if running the network
%     in GPU mode.
%
%   `learningRate`:: 1
%     Factor used to adjust a parameter's learning rate.
%
%   `weightDecay`:: 1
%     Factor used to adjust a parameter's weight decay.
%
%   `trainMethod`:: 'gradient'
%     Training method, specified as a string:
%       * 'gradient' is the default training method (SGD).
%       * 'average' uses a running average, used mostly by vl_nnbnorm.
%       * 'none' disables learning, freezing the parameter's initial value.

% Copyright (C) 2016-2017 Joao F. Henriques.
% All rights reserved.
%
% This file is part of the VLFeat library and is made available under
% the terms of the BSD license (see the COPYING file).

  properties
    value
    weightDecay
    learningRate
    trainMethod
    gpu
  end
  properties (Constant)
    trainMethods = {'gradient', 'average', 'none'}  % list of methods, see CNN_TRAIN_AUTONN
  end
  
  methods
    function obj = Param(varargin)
      opts.name = [] ;
      opts.value = 'unspecified' ;
      opts.weightDecay = 1 ;
      opts.learningRate = 1 ;
      opts.trainMethod = 'gradient' ;
      opts.gpu = true ;
      
      opts = vl_argparse(opts, varargin, 'nonrecursive') ;
      
      % validate initial value. use class(obj) in case Param is subclassed.
      assert(~isequal(opts.value, 'unspecified'), ...
        ['Must specify the VALUE property when creating a ' class(obj) '.']) ;
      assert(~isa(opts.value, 'Layer'), ...
        ['The initial value of a ' class(obj) ' cannot be a Layer.']) ;
      
      assert(any(strcmp(opts.trainMethod, obj.trainMethods)), ...
        [class(obj) '.trainMethod must be one of {' strjoin(obj.trainMethods, ', ') '}.']) ;
      
      obj.name = opts.name ;
      obj.value = opts.value ;
      obj.weightDecay = opts.weightDecay ;
      obj.learningRate = opts.learningRate ;
      obj.trainMethod = opts.trainMethod ;
      obj.gpu = opts.gpu ;
    end
    
    function displayCustom(obj, ~, ~)
      s.name = obj.name ;
      s.value = obj.value ;
      s.weightDecay = obj.weightDecay ;
      s.learningRate = obj.learningRate ;
      s.trainMethod = obj.trainMethod ;
      s.gpu = obj.gpu ;
      fprintf('Param\n\n') ;
      disp(s) ;
    end
  end
end


