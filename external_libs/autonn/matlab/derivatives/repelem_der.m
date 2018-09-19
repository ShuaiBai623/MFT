function dx = repelem_der(x,varargin)
%REPELEM_DER
%   REPELEM_DER(X, N, DZDY)
%   REPELEM_DER(X, D1, D2, ..., DZDY)
%   Derivative of REPELEM function, w.r.t. first input. Same syntax as
%   native REPELEM, plus derivative.

% Copyright (C) 2017 Joao F. Henriques, Ryan Webster
% All rights reserved.
%
% This file is part of the VLFeat library and is made available under
% the terms of the BSD license (see the COPYING file).

  dy = varargin{end};
  [x_sz, x_type] = struct_or_tensor_size(x); % handle proxy structs
  
  % NOTE: repelem and repmat have a difference in behavior
  % for vector / scalar input 
  if numel(x_sz) == 2 && any(x_sz==1) % u = repelem(v,n) behavior
    n = varargin{1};
    % repeated dim is nonsingleton dimension
    rep_dim = find(x_sz~=1);
    if rep_dim == 1
      repelem_sz = [n,1];
    else
      % rep_dim == 2 or x is scalar
      % either way the repeated dim is 2
      repelem_sz = [1,n];
    end
  else % B = repelem(A,r1,...,rN) behavior
    repelem_sz = cell2mat(varargin(1:end-1));
  end

  % find dimensions that were expanded, skip others
  nonsingleton_rep_dims = find(repelem_sz ~=1);
  for dim = nonsingleton_rep_dims
    dy_sz = size(dy);
    % split the dimension that was repeated
    split_sz = [dy_sz(1:dim-1),repelem_sz(dim),...
      dy_sz(dim)/repelem_sz(dim),dy_sz(dim+1:end)];
    % new_sz removes summed dimension
    new_sz = [dy_sz(1:dim-1),dy_sz(dim)/repelem_sz(dim),dy_sz(dim+1:end)];
    % sum the repeated elements
    dy = reshape(sum(reshape(dy,split_sz),dim),new_sz);
  end
  dx = dy;
end
