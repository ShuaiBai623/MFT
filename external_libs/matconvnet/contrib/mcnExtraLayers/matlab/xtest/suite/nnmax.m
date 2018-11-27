classdef nnmax < nntest
  methods (Test)

    function basic(test)
      batchSize = 3 ; numIn = 4 ;
      x1 = test.randn([5 5 3 batchSize]) ;
      x2 = test.randn([5 5 3 batchSize]) ;
      x3 = test.randn([5 5 3 batchSize]) ;
      x4 = test.randn([5 5 3 batchSize]) ;
      inputs = {numIn, x1, x2, x3, x4} ;
      y = vl_nnmax(inputs{:}) ;

      % check derivatives with numerical approximation
      dzdy = test.randn(size(y)) ;
      [dzdx1,dzdx2,dzdx3,dzdx4] = vl_nnmax(inputs{:}, dzdy) ;
      test.der(@(x1) vl_nnmax(numIn, x1, x2, x3, x4), ...
                     x1, dzdy, dzdx1, 1e-4*test.range) ;
      test.der(@(x2) vl_nnmax(numIn, x1, x2, x3, x4), ...
                     x2, dzdy, dzdx2, 1e-4*test.range) ;
      test.der(@(x3) vl_nnmax(numIn, x1, x2, x3, x4), ...
                     x3, dzdy, dzdx3, 1e-4*test.range) ;
      test.der(@(x4) vl_nnmax(numIn, x1, x2, x3, x4), ...
                     x4, dzdy, dzdx4, 1e-4*test.range) ;
    end

  end
end
