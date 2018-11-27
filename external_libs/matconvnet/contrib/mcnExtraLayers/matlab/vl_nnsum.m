function y = vl_nnsum(x1, x2, varargin)
% VL_NNSUM - element-wise sum of tensors X1 and X2
%   Y = VL_NNSUM(X1, X2) computes the result Y = X1 + X2
%   where X1 and X2 are HxWxCxN tensors.
%  
% Copyright (C) 2017 Samuel Albanie
% Licensed under The MIT License [see LICENSE.md for details]
  [~, dzdy] = vl_argparsepos(struct(), varargin) ;

  if isempty(dzdy)
    y = x1 + x2 ;
  else
    dzdx2 = dzdy{1} ; dzdx1 = dzdy{1} ;
    y = {dzdx1, dzdx2} ;
  end
