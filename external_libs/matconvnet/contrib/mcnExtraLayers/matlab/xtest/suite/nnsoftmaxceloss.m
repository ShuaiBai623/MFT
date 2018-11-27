classdef nnsoftmaxceloss < nntest
  methods (Test)

    function basic(test)
      x = test.randn([1 1 8 50]) ;
      p = abs(test.rand([1 1 8 50])) ;
      p = bsxfun(@rdivide, p, sum(p, 3)) ;
      y = vl_nnsoftmaxceloss(x, p) ;

      % check derivatives with numerical approximation
      dzdy = test.randn(size(y)) ;
      dzdx = vl_nnsoftmaxceloss(x, p, dzdy) ;
      test.der(@(x) vl_nnsoftmaxceloss(x, p), x, dzdy, dzdx, 1e-4*test.range) ;
    end

    function basicWithTemperature(test)
      x = test.randn([1 1 8 50]) ;
      p = abs(test.rand([1 1 8 50])) ;
      p = bsxfun(@rdivide, p, sum(p, 3)) ;
      temperature = 2 ;
      y = vl_nnsoftmaxceloss(x, p, 'temperature', temperature) ;

      % check derivatives with numerical approximation
      dzdy = test.randn(size(y)) ;
      dzdx = vl_nnsoftmaxceloss(x, p, dzdy, 'temperature', temperature) ;
      test.der(@(x) vl_nnsoftmaxceloss(x, p, 'temperature', temperature), x, ...
                                        dzdy, dzdx, 1e-3*test.range) ;
    end

    function basicWithInstanceWeights(test)
      x = test.randn([1 1 8 100]) ;
      p = abs(test.rand([1 1 8 100])) ;
      p = bsxfun(@rdivide, p, sum(p, 3)) ;
      w = abs(test.rand([1 1 1 100])) ;
      %w = w / sum(w(:)) ;
      %keyboard
      y = vl_nnsoftmaxceloss(x, p, 'instanceWeights', w) ;

      % check derivatives with numerical approximation
      dzdy = test.randn(size(y)) ;
      dzdx = vl_nnsoftmaxceloss(x, p, dzdy, 'instanceWeights', w) ;
      test.der(@(x) vl_nnsoftmaxceloss(x, p, 'instanceWeights', w), x, ...
                                        dzdy, dzdx, 1e-4*test.range) ;
    end

    function basic_softmax_logit_targets(test)
      x = test.randn([1 1 8 50]) ;
      p = abs(test.rand([1 1 8 50])) ;
      %p = bsxfun(@rdivide, p, sum(p, 3)) ;
      y = vl_nnsoftmaxceloss(x, p, 'logitTargets', true) ;

      % check derivatives with numerical approximation
      dzdy = test.randn(size(y)) ;
      dzdx = vl_nnsoftmaxceloss(x, p, dzdy, 'logitTargets', 1) ;
      test.der(@(x) vl_nnsoftmaxceloss(x, p, 'logitTargets', 1), ...
                               x, dzdy, dzdx, 1e-4*test.range) ;
    end

  end
end
