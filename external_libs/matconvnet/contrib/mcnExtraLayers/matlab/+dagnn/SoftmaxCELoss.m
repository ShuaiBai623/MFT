classdef SoftmaxCELoss < dagnn.Loss

  properties
    temperature = 1
    logitTargets = false
  end

  methods
    function outputs = forward(obj, inputs, params)
      outputs{1} = vl_nnsoftmaxceloss(inputs{1}, inputs{2}, ...
                                      'temperature', obj.temperature, ...
                                      'logitTargets', obj.logitTargets) ;
      n = obj.numAveraged ;
      m = n + size(inputs{1}, 4) ;
      obj.average = (n * obj.average + gather(outputs{1})) / m ;
      obj.numAveraged = m ;
    end

    function [derInputs, derParams] = backward(obj, inputs, params, derOutputs)
      derInputs{1} = vl_nnsoftmaxceloss(inputs{1}, inputs{2}, derOutputs{1}, ...
                                        'temperature', obj.temperature, ...
                                        'logitTargets', obj.logitTargets) ;
      derInputs{2} = [] ;
      derParams = {} ;
    end

    function obj = SoftmaxCELoss(varargin)
      obj.load(varargin) ;
      obj.temperature = obj.temperature ;
      obj.logitTargets = obj.logitTargets ;
    end
  end
end
