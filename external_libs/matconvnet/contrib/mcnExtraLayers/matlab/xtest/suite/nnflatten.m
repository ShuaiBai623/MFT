classdef nnflatten < nntest
  methods (Test)

    function basic(test)
      sampleSize = [3,3,5,10] ;
      for dim = 1:3
        sz = sampleSize ; sz(dim) = 3 ;
        x = test.randn(sz) ;
        y1 = vl_nnflatten(x, dim) ;
        test.verifyEqual(size(y1,dim), prod(sz(1:3))) ;

        % check derivatives with numerical approximation
        dzdy = test.randn(size(y1)) ;
        dzdx = vl_nnflatten(x, dim, dzdy) ;
        test.der(@(x) vl_nnflatten(x, dim), x, dzdy, dzdx, 1e-3*test.range) ;
      end
    end

  end
end
