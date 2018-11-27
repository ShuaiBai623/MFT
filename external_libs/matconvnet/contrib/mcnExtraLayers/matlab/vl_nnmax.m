function varargout = vl_nnmax(M, varargin)
% VL_NNMAX Element-wise maximum
%   Y = VL_NNMAX(M, X1, ..., XM) applies the elementwise max operator to
%   the given M input tensors, each of which should have the same input
%   dimension.
%
%   [DZDX1,..,DZDXM] = VL_NNMAX(M,X1, ...,XM, DZDY) computes the
%   derivatives of the block projected onto DZDY.
%
% Copyright (C) 2017 Samuel Albanie
% Licensed under The MIT License [see LICENSE.md for details]

  assert(M >= 2, 'at least two input tensors expected') ;
  ins = varargin(1:M) ; varargin(1:M) = [] ;
  [~, dzdy] = vl_argparsepos(struct(), varargin) ;

  shared = cat(5, ins{:}) ;
  if isempty(dzdy)
     varargout{1} = max(shared, [], 5) ;
  else
    % recompute forward max and insert derivatives at max locations
    [~,I] = max(shared, [], 5) ;
    dzdx = zeros(size(shared), 'like', shared) ;
    offsets = (I - 1) .* numel(dzdy{1}) ;
    idx = offsets(:) + (1:numel(dzdy{1}))' ;
    dzdx(idx) = dzdy{1} ;
    varargout = arrayfun(@(x) {dzdx(:,:,:,:,x)}, 1:M) ;
  end
