classdef Flatten < dagnn.Layer
  % The Flatten layer reshapes the input tensor as follows. If
  % the shape of the input is [d1 d2 d3 ... dm], then shape of the output
  % is [1, ..., 1, d1, ..., d_{a-1}, da * ... * db, d_{b+1}, ..., dm]
  % where a and b are, respectively, the first and last flattened axis.
  properties
    firstAxis = 1
    axis = 3
  end

  methods
    function outputs = forward(obj, inputs, params)
      outputs{1} = vl_nnflatten(inputs{1}, obj.axis) ;
    end

    function [derInputs, derParams] = backward(obj, inputs, params, derOutputs)
      derInputs{1} = vl_nnflatten(inputs{1}, obj.axis, derOutputs{1}) ;
      derParams = {} ;
    end
    function rfs = getReceptiveFields(obj)
      %rfs = {};
      rfs.size = [1 1] ;
      rfs.stride = [1 1] ;
      rfs.offset = [1 1] ;
      rfs = repmat(rfs, 1, 1) ;

    end

    function outputSizes = getOutputSizes(obj, inputSizes)
      szi = inputSizes{1} ;
      szi = [szi, ones(1, 4 - numel(szi))] ;
      szo = [...
        szi(1:obj.firstAxis-1), ...
        prod(szi(obj.firstAxis:obj.axis)), ...
        szi(obj.axis+1:end)] ;
      outputSizes{1} = [ones(1,numel(szi)-numel(szo)), szo] ;
    end

    function obj = Flatten(varargin)
      obj.load(varargin{:}) ;
      obj.axis = obj.axis ;
      obj.firstAxis = obj.firstAxis ;
    end
  end
end
