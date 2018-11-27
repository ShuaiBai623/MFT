function y = vl_nnmrse(x, t, varargin)
% VL_NNRMSE computes the root mean square error 
%  Y = VL_NNRMSE(X, T) computes the root mean square error between 
%  predictions X and targets T, where X and T are Nx1 arrays, and
%  Y takes a scalar value
% 
% Copyright (C) 2017 Samuel Albanie
% All rights reserved.

opts.residualScale = 1 ;
[opts, dzdy] = vl_argparsepos(opts, varargin, 'nonrecursive') ;
assert(isempty(dzdy), 'using rmse as a loss is surely madness?') 

% residuals
res = x - t ;
y = sqrt((1/numel(res)) * (opts.residualScale * res(:)).^2) ;
