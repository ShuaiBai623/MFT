function [y, dzdg, dzdb, moments] = vl_nnbrenorm_wrapper(x, g, b, ...
                                              moments, clips, test, varargin)
%VL_NNBRENORM_WRAPPER AutoNN wrapper for MatConvNet's vl_nnbrenorm
%   VL_NNBRENORM has a non-standard interface (returns a derivative for the
%   moments, even though they are not an input), so we must wrap it.
%   Layer.vl_nnbrenorm replaces a standard VL_NNBRENORM call with this one.
%
% Copyright (C) 2017 Samuel Albanie
% (based on the autonn batchnorm wrapper by Joao F. Henriques)
% All rights reserved.

[opts, dzdy] = vl_argparsepos(struct(), varargin) ;

if isscalar(g)
  g(1,1,1:size(x,3)) = g ;
end
if isscalar(b)
  b(1,1,1:size(x,3)) = b ;
end
if isscalar(moments)
  moments(1:size(x,3),1:2) = moments ;
end

if isempty(dzdy)
  y = vl_nnbrenorm(x, g, b, moments, clips, test, varargin{:}) ;
else
  [y, dzdg, dzdb, moments] = vl_nnbrenorm(x, g, b, moments, clips, ...
                                          test, dzdy{1}, varargin{2:end}) ;

  if usingDeprecatedLossFn
		moments = moments * size(x, 4) ;
  end
end

% ---------------------------------------------------------------------
function old = usingDeprecatedLossFn()
% ---------------------------------------------------------------------
  % stolen from Joao Henriques (autonn compatibility code)

  % this is needed to harmonize the behavior of two versions of vl_nnloss:
  % the legacy behavior which *sums* the loss over the batch, and the new
  % behavior that takes the *average* over the batch.
  % first, detect if the new behavior ('normalise' option) is present.
  old = false ;
  try
    vl_nnloss([], [], 'normalise', true)  ;
  catch  % unrecognized option, must be the old vl_nnloss
    old = true ;
  end
