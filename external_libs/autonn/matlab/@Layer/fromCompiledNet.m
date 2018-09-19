function netOutputs = fromCompiledNet(net)
%FROMCOMPILEDNET Decompiles a Net back into Layer objects
%   OUTPUTS = Layer.fromCompiledNet(NET) decompiles a compiled network (of
%   class Net), into their original Layer objects (i.e., a set of
%   recursively nested Layer objects).
%
%   Returns a cell array of Layer objects, each corresponding to an output
%   of the network. These can be composed with other layers, or compiled
%   into a Net object for training/evaluation.

% Copyright (C) 2017 Joao F. Henriques.
% All rights reserved.
%
% This file is part of the VLFeat library and is made available under
% the terms of the BSD license (see the COPYING file).
  
  % call a local function, to work around an occasional Matlab crash that
  % happens only for static methods
  netOutputs = fromCompiledNet_(net);
end

function netOutputs = fromCompiledNet_(net)
  % use local copies so that any changes won't reflect on the original Net
  forward = net.forward ;
  backward = net.backward ;

  % first undo ReLU short-circuiting, since it creates cycles in the graph
  [forward, backward] = undoShortCircuit(net, forward, backward);
  

  % main decompilation code.

  % create a cell array 'layers' that will contain all created Layers,
  % indexed by their output vars. e.g., if layer L outputs var V, then
  % layers{V} = L.
  assert(~isempty(net.vars)) ;
  layers = cell(size(net.vars)) ;
  
  % create Input layers
  inputNames = fieldnames(net.inputs) ;
  
  for k = 1:numel(inputNames)
    var = net.inputs.(inputNames{k}) ;  % the var index
    layers{var} = Input('name', inputNames{k}, 'gpu', net.isGpuVar(var)) ;
  end
  
  % create Param layers
  for k = 1:numel(net.params)
    p = net.params(k) ;
    
    layer = Param('name', p.name, 'value', net.vars{p.var}, ...
      'gpu', net.isGpuVar(p.var), 'learningRate', p.learningRate, ...
      'weightDecay', p.weightDecay) ;
    
    layer.trainMethod = layer.trainMethods{p.trainMethod} ;  % index to string
    layers{p.var} = layer ;
  end
  
  
  % create the layers in forward-order. this order is important, because
  % when processing a layer all its inputs must have been processed too.
  for k = 1:numel(forward)
    layer = forward(k) ;
    args = layer.args ;
    
    % replace vars in the args list with the corresponding Layer objects
    for i = 1:numel(layer.inputVars)
      assert(~isempty(layers{layer.inputVars(i)}));  % assume already created
      args{layer.inputArgPos(i)} = layers{layer.inputVars(i)} ;
    end
    
    % now create a Layer with those arguments
    obj = Layer(layer.func, args{:}) ;
    obj.name = layer.name ;
    
    % retrieve the number of input derivatives from the backward struct
    obj.numInputDer = backward(end - k + 1).numInputDer ;
    
    %retrieve precious value (set during overloaded construction)
    obj.precious = layer.precious;

    % store in the 'layers' cell array so further references to the same
    % var will fetch the same layer
    obj.name = layer.name ;
    layers{layer.outputVar(1)} = obj ;
    
    % if the layer has multiple outputs, create a Selector for each of the
    % outputs after the first one
    for i = 2:numel(layer.outputVar)
      layers{layer.outputVar(i)} = Selector(obj, i) ;
    end
  end
  
  % the last var is the root (skipping the corresponding derivative var)
  rootLayer = layers{end - 1};
  
  if ~isequal(rootLayer.func, @root)
    % single output
    netOutputs = {rootLayer} ;
  else
    % root layer's arguments are the network's outputs
    netOutputs = rootLayer.inputs ;
  end
  
  % copy meta properties to the Layers
  for o = 1:numel(netOutputs)
    netOutputs{o}.meta = net.meta ;
  end
end

function [forward, backward] = undoShortCircuit(net, forward, backward)
  % undo ReLU short-circuiting, since it creates cycles in the graph.
  % collect unused vars left behind by short-circuiting, to reuse them now.
  unused = true(size(net.vars)) ;
  unused([forward.inputVars]) = false ;
  unused([forward.outputVar]) = false ;
  unused(2:2:end) = false ;  % ignore derivative vars
  unused = find(unused) ;  % convert to list of var indexes
  
  next = 1 ;  % next unused var to take
  for k = 1:numel(forward)
    if isequal(forward(k).func, @vl_nnrelu) && ...
      forward(k).inputVars(1) == forward(k).outputVar

      % a short-circuited ReLU (output var = input var). change its output
      % to a new unused var, and update any layers that depend on it.
      oldVar = forward(k).outputVar ;
      for j = k + 1 : numel(forward)
        in = forward(j).inputVars ;
        if any(in == oldVar)  % update this dependent layer
          forward(j).inputVars(in == oldVar) = unused(next) ;
        end
      end
      
      forward(k).outputVar = unused(next) ;  % update the ReLU
      
      next = next + 1 ;
      % shouldn't happen if short-circ. ReLUs leave unused vars behind
      assert(next <= numel(unused) + 1) ;
    end
  end
end
