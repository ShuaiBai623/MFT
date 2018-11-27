classdef Normalize < dagnn.ElementWise

  properties
    channelShared = [];
    acrossSpatial = [];
  end

  methods
    function outputs = forward(obj, inputs, params)
      outputs{1} = vl_nnscalenorm(inputs{1}, params{1}) ;
    end

    function [derInputs, derParams] = backward(obj, inputs, params, derOutputs)
      ders = vl_nnscalenorm(inputs{1}, params{1}, derOutputs{1}) ;
      [derInputs, derParams] = deal(ders(1), ders(2)) ; 
    end

    function outputSizes = getOutputSizes(obj, inputSizes)
      outputSizes = inputSizes ;
    end

    function rfs = getReceptiveFields(obj)
    end

    function load(obj, varargin)
      s = dagnn.Layer.argsToStruct(varargin{:}) ;
      load@dagnn.Layer(obj, s) ;
    end

    function obj = Normalize(varargin)
      obj.load(varargin{:}) ;
    end
  end
end
