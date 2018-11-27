function y = vl_nninterp(x, shrink, zoom, varargin)
% VL_NNINTERP Dynamic bilinear interpolation
%   Y = VL_NNINTERP(X, SHRINK, ZOOM) a simple wrapper to apply bilinear
%   interpolation to achieve a given SHRINK/ZOOM factor for an
%   input tensor X in a manner that is compatible with the caffe
%   DEEPLAB framework.  X is an HxWxCxN tensor, SHRINK and ZOOM
%   are integers such that are used to define the output size.
%   The output size computation is given is best understood directly
%   from the code, and is designed to mirror the original DEEPLAB
%   implementation.
%
%   DZDX = VL_NNINTERP(X, SHRINK, ZOOM, DZDY) computes the
%   derivatives of the block projected onto DZDY.
%
%   VL_NNINTERP(..., 'option', value, ...) takes the following options:
%
%   `padBeg`:: 0
%     Padding to be applied to the top and left spatial dimensions of the
%     input tensor X before interpolation.
%
%   `padEnd`:: 0
%     Padding to be applied to the bottom and right spatial dimensions of
%     the input tensor X before interpolation.
%
% Copyright (C) 2017 Samuel Albanie
% Licensed under The MIT License [see LICENSE.md for details]

  opts.padBeg = 0 ;
  opts.padEnd = 0 ;
  [opts, dzdy] = vl_argparsepos(opts, varargin) ;

  % determine output size
  inSz = [size(x, 1) size(x, 2)] ;
  inSz = inSz + opts.padBeg + opts.padEnd ;
  outSz = ((inSz - 1) / shrink) + 1 ;
  outSz = round(outSz + (outSz -1) * (zoom - 1)) ;

  % generate sampling grid (should probably cache this)
  useGPU = isa(x, 'gpuArray') ;
  Ho = outSz(1) ; Wo = outSz(2) ;
  xi = linspace(-1, 1, Ho) ; yi = linspace(-1, 1, Wo) ;
  [yy, xx] = meshgrid(xi, yi) ;
  xxyy = [yy(:), xx(:)] ;
  if useGPU, xxyy = gpuArray(xxyy) ; end
  grid = reshape(xxyy, Wo, Ho, 2) ;
  grid = permute(grid, [3,2,1]) ;
  grid = repmat(grid, [1 1 1 size(x, 4)]) ;

 if isempty(dzdy)
   y = vl_nnbilinearsampler(x, grid) ;
 else
   y = vl_nnbilinearsampler(x, grid, dzdy{1}) ;
 end
