classdef Axpy < dagnn.Filter
  properties
    numInputs=3;
  end

  methods
    function outputs = forward(self, inputs, params)
      outputs{1} = vl_nnaxpy(inputs{1}, inputs{2}, inputs{3}) ;
    end

    function [derInputs, derParams] = backward(self, inputs, params, derOutputs)
      derInputs = vl_nnaxpy(inputs{1}, inputs{2}, inputs{3}, derOutputs{1}) ;
      derParams = {} ;
    end
    function rfs = getReceptiveFields(obj)
      numInputs = numel(obj.net.layers(obj.layerIndex).inputs) ;
      rfs.size = [1 1] ;
      rfs.stride = [1 1] ;
      rfs.offset = [1 1] ;
      rfs = repmat(rfs, numInputs, 1) ;
    end
    function kernelSize = getKernelSize(obj)
      kernelSize = obj.poolSize ;
    end

    function outputSizes = getOutputSizes(obj, inputSizes)
      %outputSizes = getOutputSizes@dagnn.Filter(obj, inputSizes) ;
      %outputSizes{1}(3) = inputSizes{1}(3) ;
      outputSizes{1} = [inputSizes{3}(1),inputSizes{3}(2),inputSizes{3}(3),inputSizes{3}(4)];
    end

    function obj = Pooling(varargin)
      obj.load(varargin) ;
    end
  end
end
