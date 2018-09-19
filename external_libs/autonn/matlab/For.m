function varargout = For(iteration, varargin)
%FOR Differentiable For-loop or recursion, with dynamic iteration count
%   [A, B, ...] = For(F, INITIAL_A, INITIAL_B, ..., NUM_ITERATIONS) ;
%   [...] = For(..., 'concatenate', DIMS) ;

% Copyright (C) 2017 Joao F. Henriques.
% All rights reserved.
%
% This file is part of the VLFeat library and is made available under
% the terms of the BSD license (see the COPYING file).


  opts.concatenate = [] ;
  [opts, initial_values] = vl_argparsepos(opts, varargin, 'nonrecursive') ;
  
  assert(numel(initial_values) >= 2, ['Must specify the initial values ' ...
    'of recursion variables, followed by the number of iterations.']) ;
  
  % extract count
  count = initial_values{end} ;
  initial_values(end) = [] ;
  
  varargout = cell(1, nargout) ;
  
  [varargout{:}] = While(iteration, initial_values{:}, 'count', count, ...
    'stopCondition', false, 'concatenate', opts.concatenate) ;

end

