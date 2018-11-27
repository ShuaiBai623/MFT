function y = vl_nnflatten(x, axis, varargin)
%VL_NNFLATTEN flatten features along axis
%   Y = VL_NNFLATTEN(X, AXIS) flattens input data X along the given 
%   axis. X is a SINGLE array of dimension H x W x C x N where (H,W) 
%   are the height and width of the features, C is the number of 
%   feature channels and N the number of of images in the batch. AXIS
%   is an integer between 1 and 3 defining the axis along which the 
%   H, W and C features will be flattened.
%
%   For example, if the AXIS is 3, then the output a is a SINGLE 
%   array of dimensions 1 x 1 x HWC x N.

%   DZDX = VL_NNFLATTEN(X, AXIS, DZDY) computes the derivatives of the
%   block projected onto DZDY. DZDX and DZDY have the same dimensions
%   as X and Y respectively.
%
%   The derivative DZDY has the same dimension of the output Y and
%   The derivative DZDX has the same dimension as the input X.
%
% Copyright (C) 2017 Samuel Albanie and Andrea Vedaldi.
% Licensed under The MIT License [see LICENSE.md for details]

  [~, dzdy] = vl_argparsepos(struct(), varargin) ;

  assert(ismember(axis, [1 2 3]), 'flatten axis must be 1, 2 or 3') ;

  sz = size(x) ; batchSize = size(x,4) ;
  outputSize = [1 1 1 batchSize] ;
  outputSize(axis) = prod(sz(1:min(length(sz),3))) ;

  if isempty(dzdy)
    y = reshape(x, outputSize) ;
  else
    y = reshape(dzdy{1}, sz) ;
  end
