classdef nnscale < nntest
  methods (Test)

    function basic(test)
      batchSize = 10 ;
      x1 = test.randn([5 5 3 batchSize]) ;
      x2 = test.randn([1 1 3 1]) ; % match along non-singletons
      b = [] ; 
      y = vl_nnscale(x1, x2, b) ;

      % check derivatives with numerical approximation
      dzdy = test.randn(size(y)) ;
      [dzdx1, dzdx2, dzdb] = vl_nnscale(x1, x2, b, dzdy) ;
      test.der(@(x1) vl_nnscale(x1, x2, b), x1, dzdy, dzdx1, 1e-3*test.range) ;
      test.der(@(x2) vl_nnscale(x1, x2, b), x2, dzdy, dzdx2, 1e-3*test.range) ;
      assert(isempty(dzdb), 'bias derivative should be empty') ;
    end

    function basicBias(test)
      batchSize = 10 ;
      x1 = test.randn([5 5 3 batchSize]) ;
      x2 = test.randn([1 1 3 1]) ; % match along non-singletons
      b = test.randn([1 1 3 1]) ; % match along non-singletons
      y = vl_nnscale(x1, x2, b) ;

      % check derivatives with numerical approximation
      dzdy = test.randn(size(y)) ;
      [dzdx1, dzdx2, dzdb] = vl_nnscale(x1, x2, b, dzdy) ;
      test.der(@(x1) vl_nnscale(x1, x2, b), x1, dzdy, dzdx1, 1e-3*test.range) ;
      test.der(@(x2) vl_nnscale(x1, x2, b), x2, dzdy, dzdx2, 1e-3*test.range) ;
      test.der(@(b) vl_nnscale(x1, x2, b), b, dzdy, dzdb, 1e-3*test.range) ;
    end

  end
end
