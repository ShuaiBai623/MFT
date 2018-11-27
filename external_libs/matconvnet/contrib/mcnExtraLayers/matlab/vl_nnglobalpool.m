function y = vl_nnglobalpool(x, varargin)
%VL_NNGLOBALPOOL CNN global poolinng.
%   Y = VL_NNGLOBALPOOL(X) applies the pooling operator to all
%   spatial locations of the data X. X is a SINGLE array of dimension 
%   H x W x C x N where (H,W) are the height and width of the map stack, 
%   C is the number of feature channels and N the number of of images 
%   in the batch.
%
%   DZDX = VL_NNGLOBALPOOL(X, POOL, DZDY) computes the derivatives of the
%   block projected onto DZDY. DZDX and DZDY have the same dimensions
%   as X and Y respectively.
%
%   VL_NNGLOBALPOOL(..., 'option', value, ...) takes the following option:
%
%   `method`:: 'avg'
%     Specify method of pooling. It can be either 'max' (retain max value
%     over all spatial locations per channel) or 'avg' (compute the average
%     value over all spatial locations per channel).
%
%   The output a is a SINGLE array of dimensions 1 x 1 x C x N.
%
%   The derivative DZDY has the same dimension of the output Y and
%   The derivative DZDX has the same dimension as the input X.
%
% Copyright (C) 2016 Samuel Albanie and Andrea Vedaldi
% Licensed under The MIT License [see LICENSE.md for details]

  opts.method = 'avg' ;
  [opts, dzdy] = vl_argparsepos(opts, varargin) ;

  if nargin <= 1 || isempty(dzdy)
    switch opts.method
      case 'avg', y = mean(mean(x, 1), 2) ;
      case 'max', y = max(max(x, [], 1), [], 2) ;
      otherwise, error('Pooling method %s not recognized', opts.method) ;            
    end
  else
    base = 1 / (size(x,1) * size(x,2)) * ones(size(x), 'like', x) ;
    y = bsxfun(@times, base, dzdy{1}) ;
  end
