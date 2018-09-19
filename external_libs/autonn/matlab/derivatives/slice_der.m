function dzdx = slice_der(x, varargin)
%SLICE_DER Derivative for the slicing operator
%   This is a dense derivative computation of slicing (e.g. x(:,:,1:3,10)).
%   A fast sparse derivative computation (e.g. for single-element accesses)
%   is implemented in Net.eval. It only calls SLICE_DER when there are
%   repeated indexes.

% Copyright (C) 2017 Joao F. Henriques.
% All rights reserved.
%
% This file is part of the VLFeat library and is made available under
% the terms of the BSD license (see the COPYING file).


  dzdy = varargin{end} ;
  subs = varargin(1:end-1) ;

  % enumerate all indexed elements explicitly to accumulate.
  % replace colon keyword/logical indexing with actual subscripts
  for i = 1:numel(subs)
    if isequal(subs{i}, ':')
      if i < numel(subs)
        subs{i} = 1:size(x,i) ;
      else  % special case, last subscripted dimension contains all trailing dimensions
        sz = size(x) ;
        subs{i} = 1:prod(sz(i:end)) ;
      end
    elseif islogical(subs{i})
      subs{i} = find(subs{i}) ;
    end
  end
  
  if isscalar(subs)
    % faster code path for linear indexes, e.g. x(:), x(1:3)
    ii = subs{1} ;
  else
    % general case, multiple subscripts
    subs_ = cell(size(subs)) ;
    [subs_{:}] = ndgrid(subs{:}) ;  % enumerate subscripts of all indexed elements
    ii = sub2ind(size(x), subs_{:}) ;  % convert to linear indexes
  end
  
  % accumulate gradients
  dzdx = accumarray(ii(:), dzdy(:), [numel(x), 1]) ;
  
  % reshape back to tensor
  dzdx = reshape(dzdx, size(x)) ;

end

