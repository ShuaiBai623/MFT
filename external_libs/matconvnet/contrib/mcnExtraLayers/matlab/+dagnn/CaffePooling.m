classdef CaffePooling < dagnn.Filter
%CAFFEPOOLING - This layer implements Caffe-style pooling, which applies
% the pooling kernel to the padding, as well as to the input (unlike matconvnet
% which does not include the padding itself in the kernel computation).
% To mimic this effect, we "pre-pad" the input using an empty convolution, then
% pass to the standard pooling layer.

  properties
    method = 'max'
    poolSize = [1 1]
    opts = {'cuDNN'}
  end

  methods
    function outputs = forward(self, inputs, params)
      outputs{1} = vl_nncaffepool(inputs{1}, self.poolSize, ...
                                  'pad', self.pad, ...
                                  'method', self.method, ...
                                  'stride', self.stride, ...
                                  'extraArgs', self.opts) ;
    end

    function [derInputs, derParams] = backward(self, inputs, params, derOutputs)
      derInputs{1} = vl_nncaffepool(inputs{1}, self.poolSize, derOutputs{1}, ...
                                  'pad', self.pad, ...
                                  'method', self.method, ...
                                  'stride', self.stride, ...
                                  'extraArgs', self.opts) ;
      derParams = {} ;
    end

    function kernelSize = getKernelSize(obj)
      kernelSize = obj.poolSize ;
    end

    function outputSizes = getOutputSizes(obj, inputSizes)
      outputSizes = getOutputSizes@dagnn.Filter(obj, inputSizes) ;
      outputSizes{1}(3) = inputSizes{1}(3) ;
    end

    function obj = CaffePooling(varargin)
      obj.load(varargin) ;
    end
  end
end
