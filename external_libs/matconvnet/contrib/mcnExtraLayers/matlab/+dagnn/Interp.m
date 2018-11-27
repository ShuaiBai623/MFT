classdef Interp < dagnn.Layer
  properties
    zoomFactor = 1
    shrinkFactor = 1
    padBeg = 0
    padEnd = 0
  end

  methods
    function outputs = forward(obj, inputs, params)
      outputs{1} = vl_nninterp(inputs{1}, obj.shrinkFactor, ...
             obj.zoomFactor, 'padBeg', obj.padBeg, 'padEnd', obj.padEnd) ;
    end

    function [derInputs, derParams] = backward(obj, inputs, params, derOutputs)
      derInputs{1} = vl_nninterp(inputs{1}, obj.shrinkFactor, ...
             obj.zoomFactor, derOutputs{1}, 'padBeg', obj.padBeg, ...
            'padEnd', obj.padEnd) ;
      derParams = {} ;
    end

    function obj = Interp(varargin)
      obj.load(varargin{:}) ;
      obj.zoomFactor = obj.zoomFactor ;
    end
  end
end
