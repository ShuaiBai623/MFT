function varargout = vl_nnwsum(varargin)
%VL_NNWSUM Differentiable weighted sum
%   Y = VL_NNWSUM(A, B, ..., 'weights', W) returns a weighted sum of
%   inputs, i.e. Y = W(1) * A + W(2) * B + ...
%
%   [DA, DB, ...] = VL_NNWSUM(A, B, ..., DZDY, 'weights', W) returns the
%   projected derivatives of the same operation with respect to all inputs,
%   except for weights W, which are assumed constant.

% Copyright (C) 2016 Joao F. Henriques.
% All rights reserved.
%
% This file is part of the VLFeat library and is made available under
% the terms of the BSD license (see the COPYING file).

  assert(numel(varargin) >= 2 && isequal(varargin{end-1}, 'weights'), ...
    'Must supply the ''weights'' property.') ;

  w = varargin{end} ;  % vector of scalar weights
  n = numel(varargin) - 2 ;
  
  if n == numel(w)
      % forward function
      y = w(1) * varargin{1} ;
      for k = 2:n
        y = bsxfun(@plus, y, w(k) * varargin{k}) ;
      end
      
      varargout = {y} ;
      
  elseif n == numel(w) + 1
      % backward function (the last argument is the derivative)
      dy = varargin{n} ;
      n = n - 1 ;
      
      varargout = cell(1, n) ;
      for k = 1:n
        dx = dy ;
        for t = 1:ndims(dy)  % sum derivatives along expanded dimensions (by bsxfun)
          [x_sz,~] = struct_or_tensor_size(varargin{k}) ; % handle proxy structs
          x_sz = [x_sz, ones(1,ndims(dy) - numel(x_sz))] ; % dims after ndims(x) are size 1
          if x_sz(t) == 1  % original was a singleton dimension
            dx = sum(dx, t) ;
          end
        end
        varargout{k} = w(k) * dx ;
      end
      
  else
    error('The number of weights does not match the number of inputs.') ;
  end
end

