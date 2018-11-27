classdef nnreshape < nntest
  methods (Test)

    function basicWithMinus(test)
      sz = [3,3,5,10] ;
      x = test.randn(sz) ;
      dim = {9,[],5} ;

      y = vl_nnreshape(x, dim) ;
      targets = [9,1,5,10] ;
      osz = size(y)  ;
      for i = 1:numel(targets)
        assert(osz(i) == targets(i)) ;
      end

      % check derivatives with numerical approximation
      dzdy = test.randn(size(y)) ;
      dzdx = vl_nnreshape(x, dim, dzdy) ;
      test.der(@(x) vl_nnreshape(x, dim), x, dzdy, dzdx, 1e-3*test.range) ;
    end

    function basicWithZero(test)
      sz = [3,3,5,10] ;
      x = test.randn(sz) ;
      dim = {[],3,5} ;

      y = vl_nnreshape(x, dim) ;
      targets = [3,3,5,10] ;
      osz = size(y)  ;
      for i = 1:numel(targets)
        assert(osz(i) == targets(i)) ;
      end

      % check derivatives with numerical approximation
      dzdy = test.randn(size(y)) ;
      dzdx = vl_nnreshape(x, dim, dzdy) ;
      test.der(@(x) vl_nnreshape(x, dim), x, dzdy, dzdx, 1e-3*test.range) ;
    end

  end
end
