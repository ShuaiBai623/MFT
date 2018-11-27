classdef nnsoftmaxt < nntest
  methods (Test)

    function basic(test)
      x = test.randn([5 5 3 3]) ;
      y = vl_nnsoftmaxt(x) ;

      % check derivatives with numerical approximation
      dzdy = test.randn(size(y)) ;
      dzdx = vl_nnsoftmaxt(x, dzdy) ;
      test.der(@(x) vl_nnsoftmaxt(x), x, dzdy, dzdx, 1e-4*test.range) ;
    end

    function basic_temp(test)
      x = test.randn([5 5 3 3]) ;
      temperature = 7 ;
      y = vl_nnsoftmaxt(x, 'temperature', temperature) ;

      % check derivatives with numerical approximation
      dzdy = test.randn(size(y)) ;
      dzdx = vl_nnsoftmaxt(x, dzdy, 'temperature', temperature) ;
      test.der(@(x) vl_nnsoftmaxt(x, 'temperature', temperature), ...
                                 x, dzdy, dzdx, 1e-4*test.range) ;
    end

    function basic_dim(test)
      x = test.randn([5 5 3 3]) ;
      dim = 1 ;
      y = vl_nnsoftmaxt(x, 'dim', dim) ;

      % check derivatives with numerical approximation
      dzdy = test.randn(size(y)) ;
      dzdx = vl_nnsoftmaxt(x, dzdy, 'dim', dim) ;
      test.der(@(x) vl_nnsoftmaxt(x, 'dim', dim), ...
                                 x, dzdy, dzdx, 1e-4*test.range) ;
    end

  end
end
