classdef nncrop_wrapper < nntest
  methods (Test)

    function basicShrink(test)
      batchSize = 10 ;
      x1 = test.randn([10 10 3 batchSize]) ;
      x2 = test.randn([5 5 3 batchSize]) ;
      crop = [1 1] ;
      y = vl_nncrop_wrapper(x1, x2, crop) ;

      % check derivatives with numerical approximation
      dzdy = test.randn(size(y)) ;
      dzdx1 = vl_nncrop_wrapper(x1, x2, crop, dzdy) ;
      test.der(@(x1) vl_nncrop_wrapper(x1, x2, crop), ...
                               x1, dzdy, dzdx1, 1e-3*test.range) ;
    end
  end
end
