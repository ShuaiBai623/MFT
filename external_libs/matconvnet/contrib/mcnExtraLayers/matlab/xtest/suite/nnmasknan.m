classdef nnmasknan < nntest
  methods (Test)

    function basic(test)
      rows = 30 ;
      cols = 10 ;
      numNaN = 50 ;
      x = test.randn([rows cols]) ;
      t = test.randn([rows cols]) ;

      p = randperm(rows*cols) ; 
      idx = p(1:numNaN) ;
      t(idx) = NaN ;

      [mx, mt] = vl_nnmasknan(x, t) ;

      % check derivatives with numerical approximation
      dzdy = test.rand(rows*cols - numNaN, 1) ;
      [dzdx,~] = vl_nnmasknan(x, t, dzdy) ;
      test.der(@(x) vl_nnmasknan(x, t), x, dzdy, dzdx, 1e-3*test.range) ;
    end
  end
end
