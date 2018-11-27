classdef L2Norm < dagnn.ElementWise

  properties
    % keep the name "param" for  compatibility with previous implementations
    param = 1e-10 ;
  end

  methods
    function outputs = forward(obj, inputs, params)
      outputs{1} = vl_nnl2norm(inputs{1}, 'epsilon', obj.param) ;
    end

    function [derInputs, derParams] = backward(obj, inputs, params, derOutputs)
      derInputs{1} = vl_nnl2norm(inputs{1}, derOutputs{1}, 'epsilon', obj.param) ;
      derParams = {} ;
    end

    function rfs = getReceptiveFields(obj)
      rfs.size = [1 1] ;
      rfs.stride = [1 1] ;
      rfs.offset = [1 1] ;
    end

    function obj = L2Norm(varargin)
      obj.load(varargin) ;
    end
  end
end
