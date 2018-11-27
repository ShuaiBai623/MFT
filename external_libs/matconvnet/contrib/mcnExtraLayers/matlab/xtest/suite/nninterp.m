classdef nninterp < nntest
  methods (Test)

    function basicShrink(test)
      zoom = 1 ;
      shrink = 3 ;
      batchSize = 10 ;
      x = test.randn([5 5 3 batchSize]) ;
      y = vl_nninterp(x, shrink, zoom) ;

      % check derivatives with numerical approximation
      dzdy = test.randn(size(y)) ;
      dzdx = vl_nninterp(x, shrink, zoom, dzdy) ;
      test.der(@(x) vl_nninterp(x, shrink, zoom), x, dzdy, dzdx, 1e-3*test.range) ;
    end

    function basicShrinkZoom(test)
      zoom = 4 ;
      shrink = 3 ;
      batchSize = 10 ;
      padBeg = 2 ;
      padEnd = 1 ;
      pad = {'padBeg', padBeg, 'padEnd', padEnd} ;
      x = test.randn([5 5 3 batchSize]) ;
      y = vl_nninterp(x, shrink, zoom, pad{:}) ;

      % check derivatives with numerical approximation
      dzdy = test.randn(size(y)) ;
      dzdx = vl_nninterp(x, shrink, zoom, dzdy, pad{:}) ;
      test.der(@(x) vl_nninterp(x, shrink, zoom, pad{:}), ...
                               x, dzdy, dzdx, 1e-3*test.range) ;
    end

  end
end
