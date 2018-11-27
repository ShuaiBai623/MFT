classdef SoftMaxTranspose < dagnn.ElementWise

  properties
      dim = 1 
  end 

  methods
    function outputs = forward(obj, inputs, params)
      outputs{1} = vl_nnsoftmaxt(inputs{1}, 'dim', obj.dim) ;
    end

    function [derInputs, derParams] = backward(obj, inputs, params, derOutputs)
      derInputs{1} = vl_nnsoftmaxt(inputs{1}, derOutputs{1}, 'dim', obj.dim) ;
      derParams = {} ;
    end

    function obj = SoftMax(varargin)
      obj.load(varargin) ;
    end
  end
end
