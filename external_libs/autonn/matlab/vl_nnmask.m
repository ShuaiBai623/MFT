function mask = vl_nnmask(x, rate, dy)
%VL_NNMASK CNN dropout mask generator
%   VL_NNMASK(X, RATE) returns a dropout mask, with dropout rate RATE.
%
%   VL_NNMASK(X, RATE, DY) returns the corresponding projected derivative,
%   which is simply DY.
%
%   This is an auxiliary function to the implementation of dropout, used by
%   Layer.vl_nndropout.

% Copyright (C) 2016-2017 Joao F. Henriques.
% All rights reserved.
%
% This file is part of the VLFeat library and is made available under
% the terms of the BSD license (see the COPYING file).

if nargin == 3
  % backward mode; the derivative of x is expected. since we only need x to
  % get its size, it doesn't change, so its derivative is the identity.
  mask = dy ;
  return
end

if nargin < 2
  rate = 0.5 ;
end

scale = 1 / (1 - rate) ;

if isa(x,'gpuArray')
  switch classUnderlying(x)
    case 'single'
      scale = single(scale) ;
    case 'double'
      scale = double(scale) ;
  end
  mask = scale * (gpuArray.rand(size(x), 'single') >= rate) ;
else
  switch class(x)
    case 'single'
      scale = single(scale) ;
    case 'double'
      scale = double(scale) ;
  end
  mask = scale * (rand(size(x), 'single') >= rate) ;
end
