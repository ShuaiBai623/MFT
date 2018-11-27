function [y, dzdg, dzdb, m] = vl_nnbrenorm(x, g, b, m, clips, test, varargin)
%VL_NNBRENORM CNN batch renormalisation.
%   Y = VL_NNBRENORM(X,G,B,M,CLIPS,TEST) applies batch renormalization
%   to the input X. Batch renormalization is defined as:
%
%      Y(i,j,k,t) = G(k) * X_HAT(i,j,k,t) + B(k)
%
%   where
%      X_HAT(i,j,k,t) = R(k) * (X_HAT(i,j,k,t) - mu(k)) / sigma(k) + D(k)
%      mu(k) = mean_ijt X(i,j,k,t),
%      sigma2(k) = mean_ijt (X(i,j,k,t) - mu(k))^2,
%      sigma(k) = sqrt(sigma2(k) + EPSILON)
%      R(k) = cutoff(sigma(k) / M(2,k)), [1/rMax, rMax])
%      D(k) = cutoff((mu(k) - M(1,k))/ M(2,k)), [-dMax, dMax])
%      rMax = clips(1)
%      dMax = clips(2)
%
%   and we define cutoff(x, [a b]) to be the operation that clips the value
%   of x to lie inside the range [a b]. The parameters G(k) and B(k) are
%   multiplicative and additive constants use to scale each data channel, M
%   is the 2xC array of moments used to track the batch mean and variance.
%   R(k) and D(k) are used to balance the current estimate of feature
%   means and variances between the statistics gathered from the current
%   minibatch, and rolling averages over previous minibatches, as discussed
%   in the paper:
%
%  `Batch Renormalization: Towards Reducing Minibatch Dependence in
%   Batch-Normalized Models` by Sergey Ioffe, 2017
%
% Copyright (C) 2017 Samuel Albanie
% All rights reserved.

  [~, dzdy] = vl_argparsepos(struct(), varargin) ;

  % unpack parameters
  epsilon = 1e-4 ; rMax = clips(1) ; dMax = clips(2) ;
  rolling_mu = permute(m(:,1), [3 2 1]) ;
  rolling_sigma = permute(m(:,2), [3 2 1]) ;

  if ~test
    % first compute statistics per channel for current minibatch and normalize
    mu = chanAvg(x) ; sigma2 = chanAvg(bsxfun(@minus, x, mu).^ 2) ;
    sigma = sqrt(sigma2 + epsilon) ;
    x_hat_ = bsxfun(@rdivide, bsxfun(@minus, x, mu), sigma) ;
    % then "renormalize"
    r = bsxfun(@min, bsxfun(@max, sigma ./ rolling_sigma, 1 / rMax), rMax) ;
    d = bsxfun(@min, bsxfun(@max, (mu - rolling_mu)./rolling_sigma,-dMax), dMax) ;
    x_hat = bsxfun(@plus, bsxfun(@times, x_hat_, r), d) ;
  else
    x_hat = bsxfun(@rdivide, bsxfun(@minus, x, rolling_mu), rolling_sigma) ;
  end

  if isempty(dzdy)
    res = bsxfun(@times, g, x_hat) ; % apply gain
    y = bsxfun(@plus, res, b) ; % add bias
  else
    % precompute some common terms
    t1 = bsxfun(@minus, x, mu) ;
    t2 = bsxfun(@rdivide, r, sigma) ;
    t3 = bsxfun(@rdivide, r, sigma2) ;
    sz = size(x) ; m = prod([sz(1:2) size(x,4)]) ;
    dzdy = dzdy{1} ; dzdx_hat = bsxfun(@times, dzdy, g) ;
    dzdsigma = chanSum(dzdx_hat .* bsxfun(@times, t1, -t3)) ;
    dzdmu = chanSum(bsxfun(@times, dzdx_hat, -t2)) ;
    t4 = bsxfun(@times, dzdx_hat, t2) + ...
         bsxfun(@times, dzdsigma,  bsxfun(@rdivide, t1, m * sigma)) ;
    dzdx = bsxfun(@plus, t4, dzdmu * (1/m)) ; y = dzdx ;
    dzdg = chanSum(x_hat .* dzdy) ; dzdb = chanSum(dzdy) ;
    m = horzcat(squeeze(mu), squeeze(sigma)) ;
  end

% -----------------------
function avg = chanAvg(x)
% -----------------------
  avg = mean(mean(mean(x, 1), 2), 4) ;

% -----------------------
function res = chanSum(x)
% -----------------------
  res = sum(sum(sum(x, 1), 2), 4) ;
