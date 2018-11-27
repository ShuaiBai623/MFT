classdef nnaxpy < nntest

  methods (Test)

    function basic(test)
      sz = [4,5,10,3] ;
      x1 = test.randn(sz) ;
      x2 = test.randn(sz) ;
      a = test.randn([1 1 sz(3) sz(4)]) ;

      y = vl_nnaxpy(a, x1, x2) ;
      dzdy = test.randn(size(y)) ;
      dzdx = vl_nnaxpy(a, x1, x2, dzdy) ;

      test.der(@(a) vl_nnaxpy(a, x1, x2), a, dzdy, dzdx{1}, 1e-3*test.range) ;
      test.der(@(x1) vl_nnaxpy(a, x1, x2), x1, dzdy, dzdx{2}, 1e-4*test.range) ;
      test.der(@(x2) vl_nnaxpy(a, x1, x2), x2, dzdy, dzdx{3}, 1e-3*test.range) ;
    end
  end
end
