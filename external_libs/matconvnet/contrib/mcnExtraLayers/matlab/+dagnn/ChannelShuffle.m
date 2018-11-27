classdef ChannelShuffle < dagnn.Layer

  properties
    group = 1
  end

  methods
    function outputs = forward(obj, inputs, params)
      outputs{1} = vl_nnchannelshuffle(inputs{1}, obj.group) ;
    end

    function [derInputs, derParams] = backward(obj, inputs, params, derOutputs)
      derInputs{1} = vl_nnchannelshuffle(inputs{1}, obj.group, derOutputs{1}) ;
      derParams = {} ;
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
