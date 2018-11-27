classdef CosineSim < dagnn.Filter

  methods
    function outputs = forward(self, inputs, params)
      outputs{1} = vl_nncosinesim(inputs{1}, inputs{2}) ;
    end

    function [derInputs, derParams] = backward(self, inputs, params, derOutputs)
      derInputs{1} = vl_nncosinesim(inputs{1}, inputs{2}, derOutputs{1}) ;
      derParams = {} ;
    end

    function outputSizes = getOutputSizes(obj, inputSizes)
      outputSizes{1} = ones(1,4) ;
      outputSizes{1}(4) = inputSizes{1}(4) ;
    end

    function obj = CosineSim(varargin)
      obj.load(varargin) ;
    end
  end
end
