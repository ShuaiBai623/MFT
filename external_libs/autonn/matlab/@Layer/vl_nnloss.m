function y = vl_nnloss(obj, varargin)
%VL_NNLOSS Additional options for vl_nnloss (standard CNN losses)
%   Y = Layer.vl_nnloss(X, C) computes the softmax-log loss of the input X,
%   with class labels C. See help vl_nnloss for more details.
%
%   This method overloads MatConvNet's vl_nnloss function for Layer
%   objects, so that instead of executing vl_nnloss, a new Layer object is
%   returned.

% Copyright (C) 2018 Joao F. Henriques.
% All rights reserved.
%
% This file is part of the VLFeat library and is made available under
% the terms of the BSD license (see the COPYING file).

  y = Layer(@vl_nnloss, obj, varargin{:}) ;
  y.numInputDer = 1 ;  % only the first derivative is defined
  
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
  
  % if the old vl_nnloss is being used, we introduce a normalization step
  if old
    y = y ./ size(obj, 4) ;
    warning('MatConvNet:NormalizedLoss', ['The most recent version of ' ...
     'vl_nnloss normalizes the loss by the batch size. The current version ' ...
     'does not. A workaround is being used, but consider updating MatConvNet.']) ;
  end
  
end
