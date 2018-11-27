function y = vl_nnreshape(x, shape, varargin)
% VL_NNRESHAPE Feature reshaping
%   Y = VL_NNRESHAPE(X, SHAPE) reshpaes the input data X to have
%   the dimensions specified by SHAPE. X is a SINGLE array of
%   dimension H x W x D x N where (H,W) are the height and width of
%   the map stack, D is the image depth (number of feature channels)
%   and N the number of of images in the stack. SHAPE is a 1 x 3 cell
%   array, the contents of which are passed in order to the MATLAB
%   reshape function. As a consequence, `[]` an be used to specify a
%   dimension which should be computed from the other two. The batch size
%   (the fourth dimension of the input) is left unchanged by this
%   reshaping operation.
%
%   Alternatively, SHAPE can be specified in the "caffe style", as an
%   array, using `0` to indicate that a dimension should be preserved and
%   `-1` to indicate a dimension that should be computed from the others
%   (this serves the same role as [] in the standard MATLAB convention).
%
%   Example:
%       Inputs: X with shape [100 100 3 5] and SHAPE = { 100 3 [] }
%       will produce an output Y with shape [100 3 100 5]
%
%   DZDX = VL_NNRESHAPE(X, SHAPE, DZDY) computes the derivatives of the
%   block projected onto DZDY. DZDX and DZDY have the same dimensions
%   as X and Y respectively.
%
% Copyright (C) 2017 Samuel Albanie and Andrea Vedaldi.
% Licensed under The MIT License [see LICENSE.md for details]

  [~, dzdy] = vl_argparsepos(struct(), varargin) ;

  if isnumeric(shape) % apply caffe style conventions if needed
    shape_ = num2cell(shape) ;
    if numel(shape_) == 2, shape_{3} = [] ; end
    k = find(shape == -1) ; if k, shape_{k} = [] ; end
    k = find(shape == 0) ;
    if k, rep = arrayfun(@(i) {size(x,i)}, k) ; shape_(k) = rep ; end
    shape = shape_ ;
  end

  batchSize = size(x, 4);

  if isempty(dzdy)
    y = reshape(x, shape{1}, shape{2}, shape{3}, batchSize) ;
  else
    y = reshape(dzdy{1}, size(x)) ;
  end
