function y = vl_nnaugdata(x, varargin)
% VL_NNAUGDATA data augmentation for visual data
%   Y = VL_NNAUGDATA(X) randomly applies a set of data augmentation
%   transformations to the HxWxCxN input tensor X to produce an
%   augmented version of the data Y (of the same shape as X).
%
%   VL_NNAUGDATA(..., 'option', value, ...) takes the following options:
%
%   `rotateLims`:: [-pi/8, pi/8]
%    Uniformly samples rotation angles (in radians) from the given range and
%    applies them to each batch element of the input.
%
%   `zoomLims' :: [0.9, 1.1]
%    Uniformly samples zoom factors from the given range and applies them to
%    each batch element of the input.
%
%   `skewLims' :: [-0.1, 0.1]
%    Uniformly samples x and x skew-factors from the given range and applies
%    them to each batch element of the input.
%
%   `randTranslation` :: true
%    If true, randomly samples an (x,y) offset for each batch element of the
%    input, taking account of the zoomScale that was applied.
%
% Copyright (C) 2018 Samuel Albanie
% All rights reserved.

  opts.rotateLims = [-pi/8, pi/8] ;
  opts.zoomLims = [0.9, 1.1] ;
  opts.skewLims = [-0.1, 0.1] ;
  opts.randTranslation = true ;
  [opts, dzdy] = vl_argparsepos(opts, varargin) ;
  assert(isempty(dzdy), 'vl_nnaugdata does not current support backprop') ;

  augs = computeAugs(numel(batch), opts) ;

% --------------------------------------------------------------------
function affs = computeAugs(batchSize, opts)
% --------------------------------------------------------------------
% Training time augmentations
  ratio = 1/25 ;
  augs = repmat(eye(3,3), 1, 1, batchSize) ;
  maxOffset = round(ratio * 224) ; % based on Zhiding Yu's paper

  minXY = randi(maxOffset, batchSize, 2) ;
  zoomSc = (1 - ratio) + (ratio*2) * rand(1, batchSize) ;
  zAffs = arrayfun(@(x) {zoomOut(zoomSc(x), minXY(x,:))}, 1:batchSize) ;
  zAffs = cat(3, zAffs{:}) ;

  vals = [-pi/18 0 pi/18] ;
  thetas = randi(3, batchSize) ;
  rAffs = arrayfun(@(x) {rotate(vals(thetas(x)))}, 1:batchSize) ;
  rAffs = cat(3, rAffs{:}) ;

  vals = [-0.1 0 0.1] ;
  skews = randi(3, batchSize, 2) ;
  sAffs = arrayfun(@(x) {skew(vals(skews(x,1)), vals(skews(x,2)))}, 1:batchSize) ;
  sAffs = cat(3, sAffs{:}) ;

  for ii = 1:batchSize
    affs(:,:,ii) = zAffs(:,:,ii) * rAffs(:,:,ii) * sAffs(:,:,ii) ;
  end

  % only augment 50% of time
  drop = find(rand(1, batchSize) > 0.5) ;
  for ii = 1:numel(drop)
    affs(:,:,drop(ii)) = eye(3,3) ;
  end

% --------------------------------------------------------
function aff = zoomOut(zoomScale, minYX)
% --------------------------------------------------------
	zs = (zoomScale - 1) / zoomScale ;
	tx = zs - 2 * zs * minYX(2) ;
	ty = zs - 2 * zs * minYX(1) ;
	aff = [ 1 0 tx ; % compute the affine matrix
					0 1 ty ;
					0 0 1] * zoomScale ;

% --------------------------------------------------------
function aff = rotate(theta)
% --------------------------------------------------------
	aff = [ cos(theta) -sin(theta) 0 ; % compute the affine matrix
					sin(theta) cos(theta) 0 ;
					0 0 1] ;

% --------------------------------------------------------
function aff = skew(s1, s2)
% --------------------------------------------------------
	aff = [ 1 s1 0 ; % compute the affine matrix
					s2 1 0 ;
					0 0 1] ;
