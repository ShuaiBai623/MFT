function Y = vl_nnsoftmaxt(x, varargin)
%VL_NNSOFTMAXT CNN softmax transpose.
%   Y = VL_NNSOFTMAXT(X) applies the softmax operator the data X. X
%   has dimension H x W x D x N, packing N arrays of W x H
%   D-dimensional vectors.
%
%   D can be thought of as the number of possible classes.  The function
%   and the function computes the softmax along the dimension specified
%   as an option.
%
%   DZDX = VL_NNSOFTMAXT(X, DZDY) computes the derivative of the block
%   projected onto DZDY. DZDX and DZDY have the same dimensions as
%   X and Y respectively.
%
%  VL_NNSOFTMAXT(.., 'option', value, ...) accepts the following options:
%
%  `dim`:: 1
%   The dimension of X along which to compute the softmax.
%
%  `temperature` :: 1
%   The temperature of the softmax
%
%  This function is based on Andrea Vedaldi's vl_nnsoftmax function.
%
% Copyright (C) 2017 Samuel Albanie
% Licensed under The MIT License [see LICENSE.md for details]

  opts.dim = 1 ;
  opts.temperature = 1 ;
  [opts, dzdy] = vl_argparsepos(opts, varargin) ;

  x = x / opts.temperature ;

  E = exp(bsxfun(@minus, x, max(x, [], opts.dim))) ;
  L = sum(E, opts.dim) ;
  Y = bsxfun(@rdivide, E, L) ;
  if isempty(dzdy)
    return
  else
    Y = (1 / opts.temperature) * Y ...
         .* bsxfun(@minus, dzdy{1}, sum(dzdy{1} .* Y, opts.dim)) ;
  end

