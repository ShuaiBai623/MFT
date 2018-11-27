function [y,dzdw] = vl_nnscalenorm(x, w, varargin)
%VL_NNSCALENORM Feature normalization with scaling.
%   Y = VL_NNSCALENORM(X, W) normalizes input features across 
%   across channels to have an L2 norm of 1, and scales each 
%   channel by a learnable weight according to the following 
%   formula:
%   
%      Y(i,j,c,n) = X(i,j,c,n) * W(1,1,c) / sum_{k=1}^C (X(i,j,k,n)^2)
%   
%   where X is an H x W x C x N 4D tensor input, Y is the 
%   H x W x C x N tensor output and W is a 1 x 1 x Q tensor of 
%   weights whose number of channels `Q` depends on the choice of
%   `channelShared` option value described below.
%
%   [DX, DW] = VL_NNSCALENORM(X, W, DY) computes the derivatives of
%   the operator projected onto P. DX and DW have the same
%   dimensions as X and W. The derivatives with respect to the weights
%   W are given by 
%   
%      dzDW(1,1,c,1) = X(i,j,k,n) / sum_{k=1}^K (X(i,j,k,n)^2)
%   
%   NOTES: This layer was introduced in the paper: "ParseNet: Looking Wider 
%   To See Better". It is useful when combining activations from different 
%   layers of the network that might possess different scales. 
%
% Copyright (C) 2017 Samuel Albanie
% Licensed under The MIT License [see LICENSE.md for details]

  [~, dzdy] = vl_argparsepos(struct(), varargin) ;

  sz = [1 1 1 1] ;
  sz(1:numel(size(x))) = size(x) ;
  multipliers = repmat(w, [sz(1:2) 1 size(x,4)]) ;

  %Set Local Response Normalization parameters 
  kappa = 0 ; alpha = 1 ; beta = 0.5 ; N = 2 * size(x,3) ;
  normParams = [N kappa alpha beta] ;
  normalizedFeats = vl_nnnormalize(x, normParams) ;

  if isempty(dzdy)
    y = multipliers .* vl_nnnormalize(x, [N kappa alpha beta]) ;
  else
    dzdw_ = dzdy{1} .* normalizedFeats ;
    dzdw = sum(sum(sum(dzdw_,1), 2), 4) ;
    dzdx = vl_nnnormalize(x, normParams, multipliers .* dzdy{1}) ;
    y = dzdx ;
  end
