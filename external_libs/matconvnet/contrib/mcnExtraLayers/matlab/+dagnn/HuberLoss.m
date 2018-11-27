classdef HuberLoss < dagnn.Loss

  properties
    sigma = 1.
  end

  methods
    function outputs = forward(obj, inputs, params)
      outputs{1} = vl_nnhuberloss(inputs{1}, inputs{2}, ...
                       'instanceWeights', inputs{3}, ...
                       obj.opts{:}) ;
      n = obj.numAveraged ;
      m = n + size(inputs{1}, 4) ;
      obj.average = (n * obj.average + gather(outputs{1})) / m ;
      obj.numAveraged = m ;
    end

    function [derInputs, derParams] = backward(obj, inputs, params, derOutputs)
      derInputs{1} = vl_nnhuberloss(inputs{1}, inputs{2}, derOutputs{1}, ...
                                'instanceWeights', inputs{3}, ...
                                obj.opts{:}) ;
      derInputs{2} = [] ;
      derInputs{3} = [] ;
      derParams = {} ;
    end

    function obj = HuberLoss(varargin)
      obj.load(varargin) ;
      obj.sigma = obj.sigma ;
    end
  end
end
