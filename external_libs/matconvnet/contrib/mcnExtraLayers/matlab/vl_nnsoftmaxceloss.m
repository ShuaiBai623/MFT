function y = vl_nnsoftmaxceloss(x, p, varargin)
%VL_NNSOFTMAXCELOSS computes the cross entropy loss from logits
%   Y = VL_NNSOFTMAXCELOSS(X, P) computes the cross entropy loss between
%   and 1 x 1 x C x N array of predicted logits and an 1 x 1 x C x N array
%   of target probabilities. Note that the third dimension of P must form
%   valid probabilities (i.e. sum to 1, and 0<=p<=1). The output Y is a
%   SINGLE scalar.
%
%   The Cross Entropy Loss between X and P is as:
%
%     Loss = - sum_n sum_c p_{cn} log(softmax(x_n)_c)
%
%   VL_NNSOFTMAXCELOSS(..., 'option', value, ...) takes the following options:
%
%   `tol` :: 1e-5
%    Used as to control the tolerance of the sanity check that ensures that
%    the target distribution P is a valid probability distribution.
%
%   `temperature`:: 1
%    The temperature of the softmax function.
%
%   `logitTargets`:: false
%    Convert p into probabilities via a softmax of the given temperature.
%
%   `instanceWeights`:: 1
%    Weights the loss contribution of each input. This can be an N x 1
%    array that weights each input individually, or a scalar (in which
%    case the same weight is applied to every input).
%
% Copyright (C) 2018 Samuel Albanie
% Licensed under The MIT License [see LICENSE.md for details]

  opts.tol = 1e-5 ;
  opts.temperature = 1 ;
  opts.instanceWeights = ones(1, 1, 1, size(x,4)) ;
  opts.logitTargets = false ;
  [opts, dzdy] = vl_argparsepos(opts, varargin) ;

  if opts.logitTargets
    p = vl_nnsoftmaxt(p, 'temperature', opts.temperature, 'dim', 3) ;
  end


  % check valid probability targets
  normCond = all(abs(sum(p,3) - 1) < opts.tol) ;
  assert(normCond, 'values of p are not correctly normalized') ;
  rangeCond = all((0 <= p(:)) & (p(:) <= 1)) ;
  assert(rangeCond, 'values of p must lie between 0 and 1') ;

  x = x / opts.temperature ;
  Xmax = max(x,[],3) ;
  ex = exp(bsxfun(@minus, x, Xmax)) ;

  if isempty(dzdy)
		t = p .* (Xmax + log(sum(ex,3)) - x) ;
    weighted = opts.instanceWeights .* t ;
    y = sum(weighted(:)) ;
  else
		q = bsxfun(@rdivide, ex, sum(ex,3)) ;
    dydx = (1 / opts.temperature) * (q - p) ;
    weighted = opts.instanceWeights .* dydx ;
		y = bsxfun(@times, dzdy{1}, weighted) ;
  end
