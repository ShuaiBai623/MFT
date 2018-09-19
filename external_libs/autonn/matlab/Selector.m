classdef Selector < Layer
%SELECTOR Selects a single output of a multiple-outputs layer
%   The Selector is used internally to implement layers with multiple
%   outputs.
%
%   Selector(LAYER, N) simply returns the Nth output of LAYER, where N > 1.
%   When a Selector is not used, a multiple-output Layer object LAYER
%   returns only its first output; a Selector allows one to retrieve the
%   other outputs.

% Copyright (C) 2016-2017 Joao F. Henriques.
% All rights reserved.
%
% This file is part of the VLFeat library and is made available under
% the terms of the BSD license (see the COPYING file).

  properties
    index
  end
  
  methods
    function obj = Selector(input, index)
      assert(index > 1, ...
        'Cannot use a Selector to get the first output of a layer (it is returned by the layer itself).') ;
      
      assert(isempty(input.numOutputs) || index <= input.numOutputs, ...
        sprintf('Attempting to get %ith output of a layer with only %i outputs.', index, input.numOutputs)) ;
      
      obj.enableCycleChecks = false ;
      obj.inputs = {input} ;
      obj.index = index ;
      obj.enableCycleChecks = true ;
    end
    
    function displayCustom(obj, varName, showLinks)
      fprintf('Selector for output #%i of layer ', obj.index) ;
      
      if ~isempty(obj.inputs{1}.name)
        label = obj.inputs{1}.name ;
      else
        label = 'inputs{1}' ;
      end
      
      if ~showLinks || isempty(varName)
        fprintf([label '\n']) ;
      else
        cmd = [varName '.inputs{1}'] ;
        fprintf('<a href="matlab:display(%s,''%s'')">%s</a>\n', cmd, cmd, label) ;
      end
    end
  end
end
