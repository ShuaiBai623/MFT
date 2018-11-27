classdef nnglobalpool < nntest
  methods (Test)

    function basic(test)
      x = test.randn([5 5 3 3]) ;
      y = vl_nnglobalpool(x, 'method', 'avg') ;

      % check derivatives with numerical approximation
      dzdy = test.randn(size(y)) ;
      dzdx = vl_nnglobalpool(x, dzdy, 'method', 'avg') ;
      test.der(@(x) vl_nnglobalpool(x, 'method', 'avg'), ...
                              x, dzdy, dzdx, 1e-3*test.range) ;
    end

  end
end
