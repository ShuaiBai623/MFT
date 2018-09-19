function [dsubs, dval] = accumarray_der(subs, val, sz, fun, dy)
%ACCUMARRAY_DER
%   [DSUBS, DVAL] = ACCUMARRAY_DER(SUBS, VAL, SZ, DY)
%   [DSUBS, DVAL] = ACCUMARRAY_DER(SUBS, VAL, SZ, FUN, DY)
%   Derivative of ACCUMARRAY function. Same syntax as native ACCUMARRAY,
%   plus derivative. DSUBS (SUBS derivative) is always 0.
%
%   Note that only FUN = [], @sum, @max and @min are supported (the first
%   two specify additive accumulation, the default). SZ must always be
%   specified, and FILLVAL is not supported yet.
%
%   See help gpuArray.accumarray for a list of supported FUN on the GPU.
%   Note that as of R2017a custom functions are not supported (and would
%   probably have poor performance due to the large number of calls).

% Copyright (C) 2017 Joao F. Henriques.
% All rights reserved.
%
% This file is part of the VLFeat library and is made available under
% the terms of the BSD license (see the COPYING file).

  dsubs = 0 ;
  if nargin < 5  % no FUN
    dy = fun ;
    fun = [] ;
  end
  
  assert(numel(val) > 1, 'ACCUMARRAY derivative for empty or scalar values is not supported.') ;
  
  % convert subscripts to linear indexes.
  % first split subs matrix into one vector per dimension, for sub2ind.
  subs_ = cell(size(subs,2), 1) ;
  for i = 1:size(subs,2)
    subs_{i} = subs(:,i) ;
  end
  I = sub2ind(sz, subs_{:}) ;
  
  
  if isempty(fun) || isequal(fun, @sum)
    % additive accumulation
    dval = dy(I) ;
    
    % reshape to tensor
    dval = reshape(dval, size(val)) ;
    
    
  elseif isequal(fun, @max) || isequal(fun, @min)
    % max/min accumulation
    
    % call accumarray again to get max/min values
    Y = accumarray(I(:), val, [prod(sz), 1], fun) ;
    
    % find inputs (val) that correspond to the max/min value
    is_max = (Y(I) == val(:)) ;
    
    dval = zeros(size(val), 'like', val);
    
    if isa(subs, 'gpuArray')
      % fast but does not break ties - when two elements equal the max/min
      % value, both will have a non-zero gradient, instead of just one.
      % this does not give the exact gradient, but is very unlikely in a
      % stochastic scenario.
      dval(is_max) = dy(I(is_max)) ;
      
    else
      % to break ties, change indexes I from RHS of the equation above to
      % the LHS. the assignment will overwrite any repeated indexes and
      % keep only the last to be assigned, e.g. X([1 1 2])=[10 20 30] gets
      % X=[20 30].
      % note this trick does not work if subs is a gpuArray, since it does
      % not support repeated indexes in an assignment.
      
      max_idx = find(is_max) ;

      % ensure first element is the last to be written (overwriting repeats)
      max_idx = max_idx(end:-1:1) ;

      % index of max input, for each output, or 0 if not present in I
      S = zeros(sz, 'like', subs) ;
      S(I(max_idx)) = max_idx ;

      % use S to pick output derivatives and copy to input-sized tensor
      valid = (S ~= 0) ;
      dval(S(valid)) = dy(valid) ;
    end
    
  else
    error('ACCUMARRAY derivative does not support function %s.', func2str(fun)) ;
  end
end

