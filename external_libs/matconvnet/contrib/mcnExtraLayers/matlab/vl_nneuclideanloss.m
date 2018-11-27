function y = vl_nneuclideanloss(x, t, varargin)
% VL_NNEUCLIDEANLOSS computes the L2 Loss
%  Y = VL_NNEUCLIDEANLOSS(X, T) computes the Euclidean Loss
%  (also known as the L2 loss) between an N x 1 array of input
%  predictions, X and an N x 1 array of targets, T. The output
%  Y is a scalar value.
%
% Copyright (C) 2017 Samuel Albanie
% All rights reserved.

  opts.instanceWeights = ones(size(x)) ;
  [opts, dzdy] = vl_argparsepos(opts, varargin, 'nonrecursive') ;

  % residuals
  res = x - t ;

  if isempty(dzdy)
    resSq = res.^2 ;
    weighted = (1/2) * opts.instanceWeights .* resSq ;
    y = sum(weighted(:)) ;
  else
    y = opts.instanceWeights .* res * dzdy{1} ;
  end
