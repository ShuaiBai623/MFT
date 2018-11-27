classdef nnnonorm < nntest

  properties (TestParameter)
    rows = {2 4}
    cols = {2 5}
    numDims = {4 8}
    batchSize = {1 2 3}
  end

  methods (Test)
    function basic(test, rows, cols, numDims, batchSize)
      H = rows ;
      W = cols ;
      C = numDims ;
      bs = batchSize ;
      x = test.randn(H, W, C, bs) ;
      g = test.randn(1, 1, C, 1) / test.range ;
      b = test.randn(1, 1, C, 1) / test.range ;

      y = vl_nnnonorm(x, g, b) ;
      dzdy = test.randn(size(y)) ;
      [dzdx,dzdg,dzdb] = vl_nnnonorm(x, g, b, dzdy) ;
      test.der(@(x) vl_nnnonorm(x, g, b), x, dzdy, dzdx, test.range * 1e-3) ;
      test.der(@(g) vl_nnnonorm(x, g, b), g, dzdy, dzdg, test.range * 1e-3) ;
      test.der(@(b) vl_nnnonorm(x, g, b), b, dzdy, dzdb, test.range * 1e-3) ;
    end
  end
end
