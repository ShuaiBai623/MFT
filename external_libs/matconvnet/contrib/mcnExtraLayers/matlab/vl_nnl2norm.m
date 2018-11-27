function y = vl_nnl2norm(x, varargin)
%VL_NNL2NORM - apply L2 normalisation to features
%  Y = VL_NNL2NORM(X) - applies L2 normalisation to each channel of the input
%  tensor X (which has shape H x W x C x N).
%
%   DZDX = VL_NNL2NORM(X, DZDY) computes the derivatives of the block
%   projected onto DZDY.
%
%   VL_NNL2NORM(..., 'option', value, ...) takes the following options:
%
%   `epsilon`:: 1e-10
%    Adds a small value to the normalisation factor (to avoid division by
%    zero).
%
%   This function is based on another funciton with the same name by
%     Subhransu Maji, Aruni RoyChowdhury, Tsung-Yu Lin
%
% Copyright (C) 2018 Samuel Albanie
% Licensed under The MIT License [see LICENSE.md for details]

  opts.epsilon = 1e-10 ;
  [opts,dzdy] = vl_argparsepos(opts, varargin) ;

  xNorm = sqrt(sum(x.*x, 3) + opts.epsilon) ;

  if isempty(dzdy)
    y = x ./ repmat(xNorm, [1, 1, size(x, 3)]) ;
  else
    dzdy = dzdy{1} ;
    A = bsxfun(@times, dzdy, xNorm.^(-1)) ;
    B = sum(x.*dzdy,3) .* xNorm.^(-3) ;
    B = bsxfun(@times, x, B) ;
    y = A - B ;
  end
