classdef EuclideanLoss < dagnn.Loss

  methods
    function outputs = forward(obj, inputs, params)
      outputs{1} = vl_nneuclideanloss(inputs{1}, inputs{2}, ...
                                      'instanceWeights', inputs{3}) ;
      n = obj.numAveraged ;
      m = n + size(inputs{1}, 4) ;
      obj.average = (n * obj.average + gather(outputs{1})) / m ;
      obj.numAveraged = m ;
    end

    function [derInputs, derParams] = backward(obj, inputs, params, derOutputs)
      derInputs{1} = vl_nneuclideanloss(inputs{1}, inputs{2}, derOutputs{1}, ...
                                        'instanceWeights', inputs{3}) ;
      derInputs{2} = [] ;
      derInputs{3} = [] ;
      derParams = {} ;
    end

    function obj = EuclideanLoss(varargin)
      obj.load(varargin) ;
    end
  end
end
