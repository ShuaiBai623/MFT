classdef ClassRecall < dagnn.Loss

  properties (Transient)
    numClasses = []
    confusion = 0
  end

  methods
    function outputs = forward(obj, inputs, params)
      [~,predictions] = max(inputs{1}, [], 3) ;
      predictions = squeeze(gather(predictions)) ;
      labels = squeeze(gather(inputs{2})) ;

      numPreds = numel(predictions) ;
      obj.confusion = obj.confusion + ...
        accumarray([labels, predictions],1,[obj.numClasses obj.numClasses]) ;

      % compute various statistics of the confusion matrix
      % GT - rows, predicted - cols
      classRec = zeros(1, obj.numClasses) ;
      for ii = 1:obj.numClasses
        tp_ = obj.confusion(ii,ii) ;
        fn_ = sum(obj.confusion(ii,:)) - tp_ ;
        classRec(ii) = tp_ / (tp_ + fn_) ;
      end

      obj.average = classRec ;
      obj.numAveraged = obj.numAveraged + numPreds ;
      outputs{1} = obj.average ;
    end

    function [derInputs, derParams] = backward(obj, inputs, params, derOutputs)
      derInputs{1} = [] ;
      derInputs{2} = [] ;
      derParams = {} ;
    end

    function reset(obj)
      obj.confusion = 0 ;
      obj.numAveraged = 0 ;
      obj.average = zeros(obj.numClasses, 1) ;
    end

    function obj = ClassRecall(varargin)
      obj.load(varargin) ;
    end
  end
end
