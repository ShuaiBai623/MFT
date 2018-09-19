function y = slice_wrapper(x, varargin)
%SLICE_WRAPPER AutoNN wrapper for Matlab's slicing operator
%   Implements the forward indexing/slicing operator as a function.

% Copyright (C) 2016-2017 Joao F. Henriques.
% All rights reserved.
%
% This file is part of the VLFeat library and is made available under
% the terms of the BSD license (see the COPYING file).

  y = x(varargin{:}) ;
end

