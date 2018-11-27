classdef nnscalenorm < nntest
  methods (Test)

    function basic(test)
      batchSize = 10 ;
      x = test.randn([5 5 3 batchSize]) ;
      w = test.randn([1 1 3 1]) ;
      y = vl_nnscalenorm(x, w) ;

      % check derivatives with numerical approximation
      dzdy = test.randn(size(y)) ;
      [dzdx,dzdw] = vl_nnscalenorm(x, w, dzdy) ;
      test.der(@(w) vl_nnscalenorm(x, w), w, dzdy, dzdw, 1e-3*test.range) ;
      test.der(@(x) vl_nnscalenorm(x, w), x, dzdy, dzdx, 1e-3*test.range) ;
    end

  end
end
