classdef nngnorm < nntest

  properties (TestParameter)
    rows = {2 4}
    cols = {2 5}
    numDims = {4 8}
    batchSize = {1 2 3}
    groups = {4} ;
  end

  methods (Test)
    function basic(test, rows, cols, numDims, batchSize, groups)
      H = rows ;
      W = cols ;
      C = numDims ;
      bs = batchSize ;
      numGroups = groups ;
      x = test.randn(H, W, C, bs) ;
      g = test.randn(1, 1, C, 1) / test.range ;
      b = test.randn(1, 1, C, 1) / test.range ;

      args = {'numGroups', numGroups} ;
      y = vl_nngnorm(x, g, b, args{:}) ;
      dzdy = test.randn(size(y)) ;
      [dzdx,dzdg,dzdb] = vl_nngnorm(x, g, b, dzdy, args{:}) ;
      test.der(@(x) vl_nngnorm(x, g, b, args{:}), x, dzdy, dzdx, test.range * 1e-3) ;
      test.der(@(g) vl_nngnorm(x, g, b, args{:}), g, dzdy, dzdg, test.range * 1e-3) ;
      test.der(@(b) vl_nngnorm(x, g, b, args{:}), b, dzdy, dzdb, test.range * 1e-3) ;
    end
  end
end
