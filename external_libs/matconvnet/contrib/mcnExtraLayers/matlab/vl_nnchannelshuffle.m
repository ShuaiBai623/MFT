function y = vl_nnchannelshuffle(x, group, varargin)
% VL_NNCHANNELSHUFFLE Channel shuffling
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
%   Example:
%       Inputs: X with shape [100 100 3 5] and SHAPE = { 100 3 [] } 
%       will produce an output Y with shape [100 3 100 5]
%
%   DZDX = VL_NNRESHAPE(X, SHAPE, DZDY) computes the derivatives of the
%   block projected onto DZDY. DZDX and DZDY have the same dimensions
%   as X and Y respectively.
%
%  This operation was originally described in:
%
%  Zhang, X., Zhou, X., Lin, M., & Sun, J. (2017). 
%  ShuffleNet: An Extremely Efficient Convolutional Neural Network for 
%  Mobile Devices. arXiv preprint arXiv:1707.01083.
%
% Copyright (C) 2017 Samuel Albanie
% Licensed under The MIT License [see LICENSE.md for details]

  [~, dzdy] = vl_argparsepos(struct(), varargin) ;
  keyboard

  batchSize = size(x, 4) ;

  if isempty(dzdy)
    y = reshape(x, shape{1}, shape{2}, shape{3}, batchSize) ;
  else
    y = reshape(dzdy{1}, size(x)) ;
  end
