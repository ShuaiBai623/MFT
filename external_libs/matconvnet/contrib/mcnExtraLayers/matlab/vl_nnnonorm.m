function [y, dzdg, dzdb] = vl_nnnonorm(x, g, b, varargin)
%VL_NNNONORM applies weights and biases, but does no normalization
%   Y = VL_NNNONORM(X,G,B) applies a set of gains and biases to
%   the input X with shape HxWxCxN. "No normalization" is defined as:
%
%      Y(i,j,k,t) = G(k') * X(i,j,k,t) + B(k')
%
%   where
%      k' = group_idx(k,C,G), where N_G is the number of groups and
%        group_idx(k,C,G) := floor(k / (C/N_G)).
%
%   VL_NNGNORM(..., 'option', value, ...) takes the following option:
%
%   This layer was largely inspired by this blog post:
%       http://www.offconvex.org/2018/03/02/acceleration-overparameterization/
%
% Copyright (C) 2018 Samuel Albanie
% All rights reserved.

  [~,dzdy] = vl_argparsepos(struct(), varargin) ;

  expectedSz = [1 1 size(x,3) 1] ;
  sg = size(g) ; sb = size(b) ;
  assert(all(expectedSz(1:numel(sg)) == sg), 'GAINS have unexpected size') ;
  assert(all(expectedSz(1:numel(sb)) == sb), 'BIASES have unexpected size') ;

  if isempty(dzdy)
    y = bsxfun(@times, g, x) ; % apply gain
    y = bsxfun(@plus, y, b) ; % add bias
  else
    dzdy = dzdy{1} ;
    dzdb = chanSum(dzdy) ;
    dzdg = chanSum(x .* dzdy) ;
    dzdx = bsxfun(@times, dzdy, g) ;
    y = dzdx ;
  end

% -----------------------
function res = chanSum(x)
% -----------------------
  res = sum(sum(sum(x, 1), 2), 4) ;
