function objs = find(obj, varargin)
%FIND Searches for layers that meet the specified criteria
%   OBJS = OBJ.FIND() returns all the Layer objects involved in the
%   computation of a Layer OBJ. This is done by recursing over OBJ's
%   inputs, collecting all the Layers found along the way. The list of
%   objects is returned in the order of a forward pass.
%
%   OBJS = OBJ.FIND(CRITERIA) only returns Layers that meet a specified
%   search criteria. The criteria can be a layer name (string), a function
%   handle, or a class name (such as 'Input' or 'Param'). A cell array is
%   returned, which may be empty if no layer meets the criteria.
%
%   OBJS = OBJ.FIND(..., N) returns only the Nth object that meets the
%   criteria, in the order of a forward pass.
%
%   This syntax is guaranteed to return a single layer, or raise an error
%   if no object is found. If N is negative, it is found in the order of a
%   backward pass (e.g. N = -1 corresponds to the last layer that meets
%   the criteria).
%
%   OBJS = OBJ.FIND(..., 'depth', D) restricts the recursion to D depth
%   levels (e.g., D = 1 means that only OBJ's direct inputs are searched).
%
%   Example:
%     images = Input() ;
%     conv1 = vl_nnconv(images, 'size', [3 3 1 100]) ;
%     relu = vl_nnrelu(conv1) ;
%     conv2 = vl_nnconv(relu, 'size', [3 3 100 10]) ;
%
%     conv2.find(@vl_nnconv)  % returns {conv1, conv2}
%     conv2.find(@vl_nnconv, 1)  % returns conv1

% Copyright (C) 2016-2017 Joao F. Henriques.
% All rights reserved.
%
% This file is part of the VLFeat library and is made available under
% the terms of the BSD license (see the COPYING file).

  % parse name-value pairs, and leave the rest in varargin
  opts.depth = inf ;
  firstArg = find(cellfun(@(s) ischar(s) && any(strcmp(s, fieldnames(opts))), varargin), 1) ;
  if ~isempty(firstArg)
    opts = vl_argparse(opts, varargin(firstArg:end), 'nonrecursive') ;
    varargin(firstArg:end) = [] ;
  end

  what = [] ;
  n = 0 ;
  if isscalar(varargin)
    if isnumeric(varargin{1})
      n = varargin{1} ;
    else
      what = varargin{1} ;
    end
  elseif numel(varargin) == 2
    what = varargin{1} ;
    n = varargin{2} ;
  elseif numel(varargin) > 3
    error('Too many input arguments.') ;
  end

  % do the work
  visited = Layer.initializeRecursion() ;
  objs = findRecursive(obj, what, n, opts.depth, visited, {}) ;

  % choose the Nth object
  if n ~= 0
    assert(numel(objs) >= abs(n), 'Cannot find a layer fitting the specified criteria.')
    if n > 0
      objs = objs{n} ;
    else
      objs = objs{numel(objs) + n + 1} ;
    end
  end
end

function selected = findRecursive(obj, what, n, depth, visited, selected)
% WHAT, N, DEPTH: Search criteria (see FIND).
% VISITED: Dictionary of objs seen during recursion so far (handle class).
% SELECTED: Cell array of selected objects.

% Copyright (C) 2016 Joao F. Henriques.
% All rights reserved.
%
% This file is part of the VLFeat library and is made available under
% the terms of the BSD license (see the COPYING file).

  
  if depth > 0
    % get indexes of inputs that have not been visited yet
    idx = obj.getNextRecursion(visited) ;
    
    % recurse on them (forward order)
    for i = idx
      selected = findRecursive(obj.inputs{i}, what, n, depth - 1, visited, selected) ;
    end
  end
  
  % add self to selected list, if it matches the pattern
  if ~visited.isKey(obj.id)  % not in the list yet
    if ischar(what)
      if any(what == '*') || any(what == '?')  % wildcards
        if ~isempty(regexp(obj.name, regexptranslate('wildcard', what), 'once'))
          selected{end+1} = obj ;
        end
      elseif isequal(obj.name, what) || isa(obj, what)
        selected{end+1} = obj ;
      end
    elseif isempty(what) || isequal(obj.func, what)
      selected{end+1} = obj ;
    end
  end
  
  % mark as seen
  visited(obj.id) = true ;
  
end

