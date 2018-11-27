function [mx, mt] = vl_nnmasknan(x, t, varargin)
%VL_NNMASKNAN masks out NaN data from targets
%   [MX, MT] = VL_NNMASKNAN(X,T) masks out NaN data present in targets
%   T and drops the corresponding entries in X.  Produces masked outputs
%   MX and MT where MT consists of the non-NaN entries of T and MX consists
%   of the corresponding entries of X. The shape of both MX and MT will be
%   Nx1, where N is the number of non-NaN entries of T.
%
%   NOTE: I think this function is borrowed from somewhere, since I don't
%   remember writing it.  Unfortunately I can't remember where it might
%   be from, making attribution difficult.
%
% Copyright (C) 2017 Samuel Albanie
% Licensed under The MIT License [see LICENSE.md for details]

  [~, dzdy] = vl_argparsepos(struct(), varargin) ;

  mask = isnan(t) ;
  mx = x ; mt = t ;

  if isempty(dzdy)
    mx(mask(:)) = [] ;
    mt(mask(:)) = [] ;
    mx = mx' ; mt = mt' ;
  else
    mx = zeros(size(x), 'like', x) ;
    mx(~mask) = dzdy{1} ;
  end
