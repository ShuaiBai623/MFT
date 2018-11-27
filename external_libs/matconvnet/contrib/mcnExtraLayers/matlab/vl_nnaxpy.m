function y = vl_nnaxpy(a, x1, x2, varargin)
% VL_NNAXPY - perform a channelwise multilpy and addition
%   Y = VL_NNAXPY(A, X1, X2) computes the result Y = A * X1 + X2
%   where X1 and X2 are HxWxCxN tensors and A is a 1x1xC array.
%
% Copyright (C) 2017 Samuel Albanie
% Licensed under The MIT License [see LICENSE.md for details]
  [~, dzdy] = vl_argparsepos(struct(), varargin) ;

  if isempty(dzdy)
    y = bsxfun(@times, a, x1) + x2 ;
  else
    dzdx2 = dzdy{1} ;
    dzdx1 = bsxfun(@times, a, dzdy{1}) ;
    dzda = sum(sum(x1 .* dzdy{1}, 1), 2) ;
    y = {dzda, dzdx1, dzdx2} ;
  end
