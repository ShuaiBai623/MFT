classdef GlobalPooling < dagnn.Filter
  properties
    method = 'avg';
    poolSize = [1 1];
  end

  methods
    function outputs = forward(self, inputs, params)
      outputs{1} = vl_nnglobalpool(inputs{1}, 'method', self.method) ;
    end

    function [derInputs, derParams] = backward(self, inputs, params, derOutputs)
      derInputs{1} = vl_nnglobalpool(inputs{1}, derOutputs{1}, 'method', self.method) ;
      derParams = {} ;
    end

    function kernelSize = getKernelSize(obj)
      kernelSize = obj.poolSize ;
    end

    function outputSizes = getOutputSizes(obj, inputSizes)
      %outputSizes = getOutputSizes@dagnn.Filter(obj, inputSizes) ;
      %outputSizes{1}(3) = inputSizes{1}(3) ;
      poolSize = [inputSizes{1}(1)  inputSizes{1}(2)];
      outputSizes{1}=[1,1,inputSizes{1}(3),inputSizes{1}(4)];
    end

    function obj = Pooling(varargin)
      obj.load(varargin) ;
    end
  end
end
