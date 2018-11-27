classdef nncosinesim < nntest

  methods (Test)

    function basic(test)
      sz = [4,5,10,6] ;
      x1 = test.randn(sz) ;
      x2 = test.randn(sz) ;

      y = vl_nncosinesim(x1, x2) ;
      dzdy = test.randn(size(y)) ;
      [dzdx1, dzdx2] = vl_nncosinesim(x1, x2, dzdy) ;

      test.der(@(x1) vl_nncosinesim(x1, x2), x1, dzdy, dzdx1, 1e-4*test.range) ;
      test.der(@(x2) vl_nncosinesim(x1, x2), x2, dzdy, dzdx2, 1e-4*test.range) ;
    end
  end
end
