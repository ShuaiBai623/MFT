classdef RMSProp < solvers.Solver
%RMSPROP RMSProp solver
%   Implements the RMSProp solver, proposed by Geoff Hinton in Lecture 6e
%   of his Coursera Class (unpublished):
%
%   http://www.cs.toronto.edu/~tijmen/csc321/slides/lecture_slides_lec6.pdf
%
%   Training is performed by calling step (see 'help solvers.Solver.step').
%   See 'autonn/examples/minimal/minimal_network.m' for a small example.
%
%   solvers.RMSProp('option', value, ...) accepts the following options:
%
%   `learningRate`:: 0.001
%     The learning rate.
%
%   `weightDecay`:: 0
%     The weight decay (regularizer).
%
%   `epsilon`:: 1e-8
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
    epsilon = 1e-8
    rho = 0.9
    
    g_sqr = {}  % squared gradient estimate
  end
  
  methods
    function o = RMSProp(varargin)
      % parse generic Solver arguments
      varargin = o.parseGenericArgs(varargin) ;
      
      % parse arguments specific to this solver
      vl_parseprop(o, varargin, {'epsilon', 'rho'}) ;
    end
    
    function w = gradientStep(o, w, dw, lr, decay)
      % use local variables for speed
      [g_sqr, epsilon, rho] = deal(o.g_sqr, o.epsilon, o.rho) ;  %#ok<*PROPLC>
      
      % initialize state variable to 0
      if isempty(g_sqr)
        g_sqr = cell(size(w)) ;
        g_sqr(:) = {0} ;
      end
      
      for i = 1:numel(w)
        % incorporate weight decay into the gradient
        grad = vl_taccum(1, dw{i}, decay(i), w{i}) ;
        
        % update squared gradient estimate
        g_sqr{i} = g_sqr{i} * rho + grad.^2 * (1 - rho) ;

        % update parameters
        w{i} = w{i} - lr(i) * grad ./ (sqrt(g_sqr{i}) + epsilon) ;
      end
      
      o.g_sqr = g_sqr ;
    end
    
    function reset(o)
      % reset state
      o.g_sqr = {} ;
    end
    
    function s = saveobj(o)
      % serialize to struct (called by the built-in function SAVE)
      % transfer state to CPU first
      s = o.saveGeneric() ;  % call parent class
      s.epsilon = o.epsilon ;
      s.rho = o.rho ;
      s.g_sqr = cellfun(@gather, o.g_sqr, 'UniformOutput', false) ;
    end
  end
  
  methods (Static)
    function o = loadobj(s)
      % deserialize from struct (called by the built-in function LOAD)
      o = solvers.RMSProp() ;
      o.epsilon = s.epsilon ;
      o.rho = s.rho ;
      o.g_sqr = s.g_sqr ;
      o.loadGeneric(s) ;  % call parent class
    end
  end
end

