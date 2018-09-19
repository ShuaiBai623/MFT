function [dop, da, db] = bsxfun_der(op, a, b, dy)
%BSXFUN_DER
%   [DOP, DA, DB] = BSXFUN_DER(OP, A, B, DY)
%   Derivative of BSXFUN function. Same syntax as native BSXFUN, plus
%   derivative. OP is a function handle (@times, @rdivide, @power).
%   Note the first output is always empty (derivative of OP).

% Copyright (C) 2016 Joao F. Henriques.
% All rights reserved.
%
% This file is part of the VLFeat library and is made available under
% the terms of the BSD license (see the COPYING file).

  if isequal(op, @times)
    da = bsxfun(@times, b, dy) ;
    if nargout > 2
      db = bsxfun(@times, a, dy) ;
    end

  elseif isequal(op, @rdivide)
    % note: @ldivide is just @rdivide with swapped inputs
    da = bsxfun(@rdivide, dy, b) ;
    if nargout > 2
      db = -dy .* bsxfun(@rdivide, a, b .^ 2) ;
    end

  elseif isequal(op, @power)
    da = dy .* a .^ (b - 1) .* b ;
    if nargout > 2
      % prevents error if log(a) becomes complex, but is not needed anyway
      % because b is constant
      db = dy .* (a .^ b) .* log(a) ;
    end

  else
    error('Derivative not implemented.') ;
  end

  % now sum derivatives along any expanded dimensions (by bsxfun)
  for t = 1:ndims(dy)  % ndims(dy) is an upper bound on ndims of a and b
    if size(a,t) == 1  % this means the original was a singleton dimension
      da = sum(da, t) ;
    end
    if nargout > 2 && size(b,t) == 1
      db = sum(db, t) ;
    end
  end
  
  dop = [] ;
end

