classdef nnbrenorm < nntest
  properties (TestParameter)
    rows = {2 8 13}
    cols = {2 8 17}
    numDims = {1 3 4}
    batchSize = {2 7}
  end
  methods (Test)
    function basic(test, rows, cols, numDims, batchSize)
      r = rows ;
      c = cols ;
      nd = numDims ;
      bs = batchSize ;
      x = test.randn(r, c, nd, bs) ; clips = [1 0] ;
      moments = test.randn(nd, 2) ;
      g = test.randn(1, 1, nd) / test.range ;
      b = test.randn(1, 1, nd) / test.range ;
      testMode = 0 ; % training mode

      y = vl_nnbrenorm(x, g, b, moments, clips, testMode) ;
      dzdy = test.randn(size(y)) ;
      [dzdx,dzdg,dzdb] = vl_nnbrenorm(x, g, b, moments, clips, testMode, dzdy) ;

      test.der(@(x) vl_nnbrenorm(x, g, b, moments, clips, testMode), ...
                           x, dzdy, dzdx, test.range * 1e-3) ;
      test.der(@(g) vl_nnbrenorm(x, g, b, moments, clips, testMode), ...
                           g, dzdy, dzdg, 1e-2) ;
      test.der(@(b) vl_nnbrenorm(x, g, b, moments, clips, testMode), ...
                           b, dzdy, dzdb, 1e-3) ;
    end
  end
end
