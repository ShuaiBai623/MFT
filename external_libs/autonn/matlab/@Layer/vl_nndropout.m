function layer = vl_nndropout(x, varargin)
%VL_NNDROPOUT Additional options for vl_nndropout (CNN dropout)
%   Y = Layer.vl_nndropout(X) applies dropout to the data X. See help
%   vl_nndropout for more details.
%
%   This method overloads MatConvNet's vl_nndropout function for Layer
%   objects, so that instead of executing vl_nndropout, a new Layer object
%   is returned.
%
%   The dropout layer is composed of a vl_nndropout_wrapper and an
%   auxiliary mask generator layer, vl_nnmask.

% Copyright (C) 2016-2017 Joao F. Henriques.
% All rights reserved.
%
% This file is part of the VLFeat library and is made available under
% the terms of the BSD license (see the COPYING file).

  opts.rate = 0.5 ;
  opts = vl_argparse(opts, varargin) ;

  % create mask generator layer
  maskLayer = Layer(@vl_nnmask, x, opts.rate) ;
  
  % create dropout wrapper layer
  layer = Layer(@vl_nndropout_wrapper, x, maskLayer, Input('testMode')) ;
  
  % vl_nndropout_wrapper doesn't return a derivative for the mask
  layer.numInputDer = 1 ;

end

