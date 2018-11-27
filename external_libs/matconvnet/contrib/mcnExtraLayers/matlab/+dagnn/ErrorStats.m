classdef ErrorStats < dagnn.Loss

  properties
    numClasses = []
  end

  properties (Transient)
    confusion = 0
    classDist = 0 ;
  end

  methods
    function outputs = forward(obj, inputs, params)
      if isempty(obj.numClasses) % :( desperate times
        obj.numClasses = 8 ; obj.average = zeros(1, 8) ;
      end

      [~,predictions] = max(inputs{1}, [], 3) ;
      predictions = squeeze(gather(predictions)) ;
      labels = squeeze(gather(inputs{2})) ;
      obj.confusion = obj.confusion + ...
        accumarray([labels, predictions], 1, [obj.numClasses obj.numClasses]) ;
      classDist = sum(obj.confusion, 2)' ;
      classAccs = diag(obj.confusion)' ;
      outputs{1} = classAccs ; % perform normalisation later
      obj.accumulateAverage(classAccs, classDist)  ;
    end

    function [derInputs, derParams] = backward(obj, inputs, params, derOutputs)
      derInputs{1} = [] ;
      derInputs{2} = [] ;
      derParams = {} ;
    end

    function accumulateAverage(obj, classAccs, classDist)
      if obj.ignoreAverage, return ; end
      n = obj.numAveraged ;
      m = max(n + classDist, 1e-5) ;
      classAvgs = bsxfun(@plus, n .* obj.average, classAccs) ./ m ;
      obj.average = classAvgs ;
      obj.classDist = obj.classDist + classDist ;
      obj.numAveraged = m ;
    end

    function reset(obj)
      obj.confusion = 0 ;
      obj.numAveraged = 0 ;
      obj.average = zeros(1, obj.numClasses) ;
    end

    function obj = ErrorStats(varargin)
      obj.load(varargin) ;
    end
  end
end
