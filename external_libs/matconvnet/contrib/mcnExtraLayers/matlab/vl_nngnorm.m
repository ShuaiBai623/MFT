function [y, dzdg, dzdb] = vl_nngnorm(x, g, b, varargin)
%VL_NNGNORM CNN group normalization.
%   Y = VL_NNGNORM(X,G,B) applies group normalization
%   to the input X with shape HxWxCxN. Group normalization is defined as:
%
%      Y(i,j,k,t) = G(k',t) * X_HAT(i,j,k,t) + B(k',t)
%
%   where
%      k' = group_idx(k,C,G), where N_G is the number of groups and
%        group_idx(k,C,G) := floor(k / (C/N_G)).
%      X_HAT(i,j,k,t) = (X_HAT(i,j,k,t) - mu(k',t)) / sigma(k',t)
%      mu(k',t) = mean_ijk'' X(i,j,k'',t),
%      sigma2(k',t) = mean_ijk'' (X(i,j,k'',t) - mu(k'',t))^2,
%      sigma(k',t) = sqrt(sigma2(k) + EPSILON)
%        where k'' takes values such that group_idx(k'',C,G) == group_idx(k,C,G)
%
%   VL_NNGNORM(..., 'option', value, ...) takes the following option:
%
%   `numGroups`:: 32
%    The number of groups used to split the channels when computing
%    normalization statistics.
%
%   `epsilon`:: 1e-4
%    A parameter to add stability to the normalization operation.
%
%   Notes: GroupNorm is introduced in the paper:
%      `Group Normalization, Yuxin Wu, Kaiming He,
%      arXiv preprint arXiv:1803.08494 (2018)
%
% Copyright (C) 2018 Samuel Albanie
% All rights reserved.

  opts.numGroups = 32 ;
  opts.epsilon = 1e-4 ;
  [opts,dzdy] = vl_argparsepos(opts, varargin) ;

  bsize = size(x, 4) ;
  expectedSz = [1 1 size(x,3) 1] ;
  sg = size(g) ; sb = size(b) ;
  assert(all(expectedSz(1:numel(sg)) == sg), 'GAINS have unexpected size') ;
  assert(all(expectedSz(1:numel(sb)) == sb), 'BIASES have unexpected size') ;

  szX = size(x) ; % store original shape

  % compute statistics per group for current minibatch and normalize
  x = reshape(x, size(x,1), size(x,2), [], opts.numGroups, bsize) ;

  mu = groupAvg(x) ;
  sigma2 = groupAvg(bsxfun(@minus, x, mu).^ 2) ;
  sigma = sqrt(sigma2 + opts.epsilon) ;
  x_hat = bsxfun(@rdivide, bsxfun(@minus, x, mu), sigma) ;

  if isempty(dzdy)
    x_hat_ = reshape(x_hat, szX) ;
    y = bsxfun(@times, g, x_hat_) ; % apply gain
    y = bsxfun(@plus, y, b) ; % add bias
  else
    dzdy = dzdy{1} ;
    dzdb = chanSum(dzdy) ;
    x_hat_ = reshape(x_hat, szX) ; dzdg = chanSum(x_hat_ .* dzdy) ;
    dzdy = reshape(dzdy, size(x,1), size(x,2), [], opts.numGroups, bsize) ;

    g_ = reshape(g, 1, 1, size(dzdy, 3), []) ;
    dzdx_hat = bsxfun(@times, dzdy, g_) ;
    t1 = bsxfun(@minus, x, mu) ;
    m = prod([size(x,1) size(x,2) size(x,3)]) ;
    dzdsigma = groupSum((-1/2) * dzdx_hat .* bsxfun(@rdivide, t1, sigma.^3)) ;

    dzdmu = groupSum(bsxfun(@rdivide, dzdx_hat, -sigma)) + ...
                bsxfun(@times, dzdsigma, -2 * groupAvg(t1)) ;

    t4 = bsxfun(@rdivide, dzdx_hat, sigma) + ...
         bsxfun(@times, dzdsigma,  (2 / m) * t1) ;
    dzdx = bsxfun(@plus, t4, dzdmu * (1/m)) ;
    y = reshape(dzdx, szX) ;
  end

% ----------------------------------------
function avg = groupAvg(x)
% ----------------------------------------
  avg = mean(mean(mean(x, 1), 2), 3) ;

% -----------------------
function res = groupSum(x)
% -----------------------
  res = sum(sum(sum(x, 1), 2), 3) ;

% -----------------------
function res = chanSum(x)
% -----------------------
  res = sum(sum(sum(x, 1), 2), 4) ;
