classdef Input < Layer
%INPUT Defines a network input (such as images or labels)
%   The Input class defines a network input, and is generally the starting
%   point for building a network. See 'help Layer' for more.
%
%   Input('option', value, ...) accepts the following options:
%
%   `name`:: []
%     Specifies the layer name. Inputs can be created with no arguments,
%     leaving the name property empty, but it must be filled in before
%     compilation. This can be done manually (obj.name = 'name'), or
%     automatically with Layer.workspaceNames() or obj.sequentialNames().
%
%   `gpu`:: false
%     Marks the Input as containing a GPU array. This means that Net.eval
%     will automatically convert it to a gpuArray, if running in GPU mode.

% Copyright (C) 2016-2017 Joao F. Henriques.
% All rights reserved.
%
% This file is part of the VLFeat library and is made available under
% the terms of the BSD license (see the COPYING file).

  properties
    gpu
  end
  
  methods
    function obj = Input(varargin)
      opts.name = [] ;
      opts.gpu = false ;
      
      if isscalar(varargin) && ischar(varargin{1})
        % special syntax, just pass in the name
        opts.name = varargin{1} ;
      else
        opts = vl_argparse(opts, varargin, 'nonrecursive') ;
      end
      
      obj.name = opts.name ;
      obj.gpu = opts.gpu ;
    end
    
    function displayCustom(obj, ~, ~)
      s.name = obj.name ;
      s.gpu = obj.gpu ;
      fprintf('Input\n\n') ;
      disp(s) ;
    end
  end
end
