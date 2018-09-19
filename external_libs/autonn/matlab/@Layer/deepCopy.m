function other = deepCopy(obj, varargin)
%DEEPCOPY Copies a network or subnetwork, optionally sharing some layers
%   OTHER = OBJ.DEEPCOPY() returns a deep copy of the network (or
%   subnetwork) that has output OBJ. This essentially recurses through all
%   the input layers of OBJ, copying them and all of their inputs as well.
%
%   To create a shallow copy (i.e., not copy inputs recursively), use
%   OTHER = OBJ.COPY().
%
%   OBJ.DEEPCOPY(..., 'option', value, ...) accepts the following options:
%
%   `share`:: {}
%     Excludes a set of layers from the deep copy, given as a cell array.
%     The layers will be shared between the original network and the copied
%     network. This can be used to implement shared parameters and shared
%     inputs (e.g. for two-stream networks), or to define the boundaries of
%     the deep copy (to only copy up to a certain layer).
%
%   `renameFn`:: @deal (no renaming)
%     Allows renaming the layers, by passing the layer names through the
%     given function handle. For example, to append the prefix 'streamA_'
%     to every copied layer, use: 'renameFn', @(name) ['streamA_' name].
%     Note that special inputs, such as Input('name', 'testMode'), will not
%     be renamed, because that would change their behavior.
%
%   `copyFn`:: [] (simple copy)
%     Specifies a different function handle to create a shallow copy of
%     each layer. Its input argument is a Layer and its output must be a
%     copy of that layer. This allows more complex editing of layers during
%     the copy operation (e.g. changing a layer type to another type).

% Copyright (C) 2016-2017 Joao F. Henriques.
% All rights reserved.
%
% This file is part of the VLFeat library and is made available under
% the terms of the BSD license (see the COPYING file).

  opts.share = {} ;
  opts.renameFn = @deal ;
  opts.copyFn = [] ;
  opts = vl_argparse(opts, varargin, 'nonrecursive') ;

  % map between (original) object ID and its copied instance. also acts as
  % a 'visited' list, to avoid redundant recursions.
  visited = Layer.initializeRecursion() ;
  
  shared = opts.share;
  if ~iscell(shared)
    shared = {shared} ;
  end
  
  for i = 1:numel(shared)
    assert(isa(shared{i}, 'Layer'), 'The option ''share'' must specify a cell array of Layer objects.') ;
    assert(~eq(shared{i}, obj, 'sameInstance'), 'The root layer of a deep copy cannot be a shared layer.') ;
    
    % propagate shared status (see below)
    shareRecursive(shared{i}, visited) ;
  end

  % do the actual copy
  other = deepCopyRecursive(obj, opts.copyFn, opts.renameFn, visited) ;
end

function shareRecursive(shared, visited)
  % shared layers are just considered visited/copied, pointing to
  % themselves as the new copy.
  visited(shared.id) = shared ;
  
  % propagate shared status: any layer that this layer depends on must also
  % be shared. otherwise, a shared layer would be depending on both the
  % original layer and its copy; a contradiction that leads to subtle bugs.
  for i = 1:numel(shared.inputs)
    in = shared.inputs{i} ;
    if isa(in, 'Layer') && ~visited.isKey(in.id)
      shareRecursive(in, visited) ;
    end
  end
end

function other = deepCopyRecursive(original, copyFn, renameFn, visited)
  % create a shallow copy first
  if isempty(copyFn)
    other = original.copy() ;
  else
    other = copyFn(original) ;
  end
  
  % rename if necessary
  if ~isa(other, 'Input') || ~strcmp(other.name, 'testMode')
    other.name = renameFn(other.name) ;
  end

  % pointer to the copied object, to be reused by any subsequent deep
  % copied layer that refers to the original object. this also marks it
  % as seen during the recursion.
  visited(original.id) = other ;

  % recurse on inputs
  for i = 1:numel(other.inputs)
    in = other.inputs{i} ;
    if isa(in, 'Layer')
      in.enableCycleChecks = false ;  % prevent cycle check when modifying a layer's input
      
      if visited.isKey(in.id)  % already seen/copied this original object
        other.inputs{i} = visited(in.id) ;  % use the copy
      else  % unseen/uncopied object, recurse on it and use the new copy
        other.inputs{i} = deepCopyRecursive(in, copyFn, renameFn, visited) ;
      end
      
      in.enableCycleChecks = true ;
    end
  end
end

