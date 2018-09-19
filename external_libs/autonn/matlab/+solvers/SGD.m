classdef SGD < solvers.Solver
%SGD Stochastic Gradient Descent solver
%   Implements a Stochastic Gradient Descent solver with momentum.
%
%   Training is performed by calling step (see 'help solvers.Solver.step').
%   See 'autonn/examples/minimal/minimal_network.m' for a small example.
%
%   solvers.SGD('option', value, ...) sets the following properties:
%
%   `learningRate`:: 0.001
%     The learning rate.
%
%   `weightDecay`:: 0
%     The weight decay (regularizer).
%
%   `momentum`:: 0.9
%     The amount of momentum (0 to disable).

% Copyright (C) 2018 Joao F. Henriques.
% All rights reserved.
%
% This file is part of the VLFeat library and is made available under
% the terms of the BSD license (see the COPYING file).

  properties
    momentum = 0.9
    state = {}  % momentum tensors
  end
  
  methods
    function o = SGD(varargin)
      % parse generic Solver arguments
      varargin = o.parseGenericArgs(varargin) ;
      
      % parse arguments specific to this solver
      vl_parseprop(o, varargin, {'momentum'}) ;
    end
    
    function w = gradientStep(o, w, dw, lr, decay)
      % use local variables for speed
      momentum = o.momentum ;  %#ok<*PROPLC>
      state = o.state ;
      
      % initialize momentum state to 0
      if isempty(state)
        state = cell(size(w)) ;
        state(:) = {0} ;
      end
      
      for i = 1:numel(w)
        % incorporate weight decay into the gradient
        grad = vl_taccum(1, dw{i}, decay(i), w{i}) ;
        
        % update momentum
        state{i} = vl_taccum(momentum, state{i}, -1, grad) ;

        % update parameters
        w{i} = vl_taccum(1, w{i}, lr(i), state{i}) ;
      end
      
      o.state = state ;
    end
    
    function reset(o)
      % reset state
      o.state = {} ;
    end
    
    function s = saveobj(o)
      % serialize to struct (called by the built-in function SAVE)
      % transfer state to CPU first
      s = o.saveGeneric() ;  % call parent class
      s.momentum = o.momentum ;
      s.state = cellfun(@gather, o.state, 'UniformOutput', false) ;
    end
  end
  
  methods (Static)
    function o = loadobj(s)
      % deserialize from struct (called by the built-in function LOAD)
      o = solvers.SGD() ;
      o.momentum = s.momentum ;
      o.state = s.state ;
      o.loadGeneric(s) ;  % call parent class
    end
  end
end

