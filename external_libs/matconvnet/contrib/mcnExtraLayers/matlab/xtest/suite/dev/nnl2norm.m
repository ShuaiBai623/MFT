classdef nnl2norm < nntest
  methods (Test)

    function basic(test)
      x = test.randn([5 5 3 3]) ;
      y = vl_nnl2norm(x) ;

      % check derivatives with numerical approximation
      dzdy = test.randn(size(y)) ;
      dzdx = vl_nnl2norm(x, dzdy) ;
      test.der(@(x) vl_nnl2norm(x), x, dzdy, dzdx, 1e-3*test.range) ;
    end

  end
end
