function [y,dzdx2,dzdb] = vl_nnscale(x1, x2, b, varargin)
% VL_NNSCALE Apply feature scale and offset
%   Y = VL_NNSCALE(X1, X2, B) rescales array X1 by array X2 and adds a
%   a bias B where X1 has shape H1 x W1 x C1 x N, X2 has shape
%   H2 x W2 x C2 x N and B has shape H3 x W3 x C3.  These shapes
%   must match along non-singleton dimensions. The bias term B may
%   also be an empty array, in which case no offset is applied.
%
%   [DZDX1, DZDX2, DZDB] = VL_NNSCALE(X1, X2, B, DZDY) computes the
%   derivatives of the block projected onto DZDY. DZDX1, DZDX2 and
%   DZDB have the same dimensions as X1, X2 and B respectively.
%
% Copyright (C) 2017 Samuel Albanie.
% Licensed under The MIT License [see LICENSE.md for details]

  opts.size = [0 0 0 0] ;
  [~, dzdy] = vl_argparsepos(opts, varargin) ;

  if isempty(dzdy)
    y = bsxfun(@times, x1, x2) ;
    if ~isempty(b), y = bsxfun(@plus, y, b) ; end
  else
    dzdx1 = bsxfun(@times, dzdy{1}, x2) ;
    dzdx2 = bsxfun(@times, dzdy{1}, x1) ;
    % perform some magic to "undo" the bsxfun effect for the derivative
    % (NOTE: here we are assuming that expanded singletons lie in x2,
    % -> this should be addressed)
    msg = 'this implementation expects any singletons to occur in X2, not X1' ;
    sz1 = size(x1) ; sz2 = size(x2) ; n = min(numel(sz1), numel(sz2)) ;
    assert(all(sz2(1:n) <= sz1(1:n)), msg) ;
    sz = [sz2 1 1 1 1] ; sz = sz(1:4) ;
    for k = find(sz == 1), dzdx2 = sum(dzdx2, k) ; end

    if ~isempty(b)
      dzdb = dzdy{1} ; szb = size(b) ; n = numel(szb) ;
      sz = [1 1 1 1] ; sz(1:n) = szb ; % handle matlab dropping of singletons
      for k = find(sz == 1), dzdb = sum(dzdb, k) ; end
    else
      dzdb = [] ;
    end
    y = dzdx1 ;
  end
