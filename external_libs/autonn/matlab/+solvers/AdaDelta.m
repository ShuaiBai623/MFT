classdef AdaDelta < solvers.Solver
%ADADELTA AdaDelta solver
%   Implements the AdaDelta solver, proposed in:
%
%     Zeiler, "AdaDelta: An Adaptive Learning Rate Method", arXiv preprint,
%     2012.
%
%   AdaDelta sets its own learning rate, so any learning rate setting will
%   be ignored.
%
%   Training is performed by calling step (see 'help solvers.Solver.step').
%   See 'autonn/examples/minimal/minimal_network.m' for a small example.
%
%   solvers.AdaDelta('option', value, ...) accepts the following options:
%
%   `weightDecay`:: 0
%     The weight decay (regularizer).
%
%   `epsilon`:: 1e-6
%      Small additive constant to regularize variance estimate.
%
%   `rho`:: 0.9
%      Moving average window for variance update, between 0 and 1 (larger
%      values result in slower/more stable updating).

% Copyright (C) 2016-2018 Joao F. Henriques.
% All rights reserved.
%
% This file is part of the VLFeat library and is made available under
% the terms of the BSD license (see the COPYING file).

  properties
    epsilon = 1e-6
    rho = 0.9
    
    g_sqr = {}  % squared gradient estimate
    delta_sqr = {}  % squared delta estimate
  end
  
  methods
    function o = AdaDelta(varargin)
      if any(strcmp(varargin, 'learningRate'))
        warning('AutoNN:AdaDeltaLR', 'AdaDelta does support learning rates.') ;
      end
      
      % parse generic Solver arguments
      varargin = o.parseGenericArgs(varargin) ;
      
      % parse arguments specific to this solver
      vl_parseprop(o, varargin, {'epsilon', 'rho'}) ;
    end
    
    function w = gradientStep(o, w, dw, ~, decay)
      % use local variables for speed
      [g_sqr, delta_sqr, epsilon, rho] = deal(o.g_sqr, o.delta_sqr, o.epsilon, o.rho) ;  %#ok<*PROPLC>
      
      % initialize all state variables to 0
      if isempty(g_sqr)
        g_sqr = cell(size(w)) ;
        g_sqr(:) = {0} ;
        delta_sqr = g_sqr ;
      end
      
      for i = 1:numel(w)
        % incorporate weight decay into the gradient
        grad = vl_taccum(1, dw{i}, decay(i), w{i}) ;
        
        % update squared gradient estimate
        g_sqr{i} = g_sqr{i} * rho + grad.^2 * (1 - rho) ;
        
        % compute delta
        delta = -sqrt((delta_sqr{i} + epsilon)) ./ ...
                 sqrt((g_sqr{i} + epsilon)) .* grad ;

        % update squared delta estimate
        delta_sqr{i} = delta_sqr{i} * rho + delta.^2 * (1 - rho) ;

        % update parameters
        w{i} = w{i} + delta ;
      end
      
      o.g_sqr = g_sqr ;
      o.delta_sqr = delta_sqr ;
    end
    
    function reset(o)
      % reset state
      o.g_sqr = {} ;
      o.delta_sqr = {} ;
    end
    
    function s = saveobj(o)
      % serialize to struct (called by the built-in function SAVE)
      % transfer state to CPU first
      s = o.saveGeneric() ;  % call parent class
      s.epsilon = o.epsilon ;
      s.rho = o.rho ;
      s.g_sqr = cellfun(@gather, o.g_sqr, 'UniformOutput', false) ;
      s.delta_sqr = cellfun(@gather, o.delta_sqr, 'UniformOutput', false) ;
    end
  end
  
  methods (Static)
    function o = loadobj(s)
      % deserialize from struct (called by the built-in function LOAD)
      o = solvers.AdaDelta() ;
      o.epsilon = s.epsilon ;
      o.rho = s.rho ;
      o.g_sqr = s.g_sqr ;
      o.delta_sqr = s.delta_sqr ;
      o.loadGeneric(s) ;  % call parent class
    end
  end
end

