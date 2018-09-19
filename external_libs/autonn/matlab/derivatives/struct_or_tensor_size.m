function [x_sz,x_type] = struct_or_tensor_size(x)
% STRUCT_OR_TENSOR_SIZE
%   STRUCT_OR_TENSOR_SIZE is used for nonprecious derivatives
%
%   Some layers replace their input on the forward pass
%   with a struct containing the size and type. This function
%   returns the size and a dummy type casted variable, regardless
%   of if x is a struct or a tensor. 

% Copyright (C) 2017 Joao Henriques, Ryan Webster
% All rights reserved.
%
% This file is part of the VLFeat library and is made available under
% the terms of the BSD license (see the COPYING file).

  if isstruct(x)
    x_sz = x.size;
    x_type = x.type; % var type casted as original x
  else
    x_sz = size(x);
    x_type = x; % just return the tensor
  end
end