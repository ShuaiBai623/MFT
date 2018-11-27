function [y,dzdx2] = vl_nnscosinesim(x1, x2, varargin)
% VL_NNCOSINESIM Compute the cosine similariy between features
%   Y = VL_NNCOSINESIM(X1, X2) computes the cosine similarity between X1
%   and X2 where X1 has shape H x W x C x N, X2 has shape
%   H x W x C x N and Y is has shape 1 x 1 x 1 x N (producing a scalar
%   per batch element).
%
%   [DZDX1, DZDX2] = VL_NNCOSINESIM(X1, X2, DZDY) computes the
%   derivatives of the block projected onto DZDY. DZDX1 and DZDX2
%   have the same dimensions as X1, X2 and B respectively.
%
%   NOTES: The cosine similarity is defined between vectors A and B of
%   dimension N x 1 to be the scalar given by:
%   cos_sim = a'b / (norm(a,2) * norm(b,2))
%   To operate with tensors, the first three channels (H, W and C) are
%   treated as the vector elements, while the fourth dimension denotes
%   batch indicies.
%
%   VL_NNCOSINELOSS(.., 'option', value, ...) accepts the following options:
%
%   `eps`:: 1e-6
%    A small value to avoid possible division by zero.
%
% Copyright (C) 2018 Samuel Albanie.
% Licensed under The MIT License [see LICENSE.md for details]

  opts.eps = 1e-6 ;
  [~, dzdy] = vl_argparsepos(struct(), varargin) ;

  sz1 = size(x1) ; sz2 = size(x2) ;
  assert(all(sz1 == sz2), 'tensor sizes do not match') ;
  x1 = reshape(x1, [], sz1(4)) ;
  x2 = reshape(x2, [], sz1(4)) ;
  dots = dot(x1, x2, 1) ;
  x1_norms = sqrt(sum(x1 .* x1, 1)) ;
  x2_norms = sqrt(sum(x2 .* x2, 1)) ;

  if isempty(dzdy)
    y = dots ./ max(x1_norms .* x2_norms, opts.eps) ;
    y = reshape(y, 1, 1, 1, sz1(4)) ;
  else
    dzdy = dzdy{1} ; dsize = size(dzdy) ;
    assert(numel(dsize) == 4 & all(dsize(1:3) == ones(1,3)) ...
           & dsize(4) == sz1(4), 'DZDY has an unexpected size') ;
    t1 = bsxfun(@rdivide, x2, max(x1_norms .* x2_norms, opts.eps)) ;
    t2 = bsxfun(@rdivide, bsxfun(@times, x1, dots), ...
                              max(x1_norms .^3 .* x2_norms, opts.eps)) ;
    dzdx1 = t1 - t2 ;

    t1 = bsxfun(@rdivide, x1, max(x2_norms .* x1_norms, opts.eps)) ;
    t2 = bsxfun(@rdivide, bsxfun(@times, x2, dots), ...
                              max(x2_norms .^3 .* x1_norms, opts.eps)) ;
    dzdx2 = t1 - t2 ;

    % reshape to match input and compute projected derivatives
    dzdx1 = reshape(dzdx1, sz1) ;
    dzdx2 = reshape(dzdx2, sz2) ;
    dzdx1 = bsxfun(@times, dzdx1, dzdy) ;
    dzdx2 = bsxfun(@times, dzdx2, dzdy) ;
    y = dzdx1 ;
  end
