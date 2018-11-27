classdef nntukeyloss < nntest
  methods (Test)

    function basic(test)
      % We have to be a little bit devious when constructing the 
      % numerical check - if computing
      %  x(i) + delta 
      % changes the value of the median of the residuals, the MAD value
      % will also change and there will appear to be a discontinuity
      % Therefore, we only run derivative checks on part of the input
      % In particular, we deliberately run checks on a portion 
      % the inputs (which should not trigger the change in median)
      n = 50 ;
      safety = 5 ;
      m = n * safety ;

      % create extra entries to safeguard the median
      xSource = sort(test.randn([m 1]) / 1) ;
      x = xSource(1:n) ;
      xPad = xSource(n+1:end) ;
      fullX = [x ; xPad] ;

      t = sort(test.randn([m 1]) / 1) ;
      y = vl_nntukeyloss(fullX, t) ;

      % check derivatives with numerical approximation
      dzdy = test.randn(size(y)) ;
      dzdx = vl_nntukeyloss(fullX, t, dzdy) ;

      % restrict to test range
      dzdx = dzdx(1:numel(x)) ;
      test.der(@(x) splitInputsTukey(x, t, xPad), x, dzdy, dzdx, 1e-3*test.range) ;
    end

  end
end

% ---------------------------------------
function y = splitInputsTukey(x, t, xPad)
% ---------------------------------------

fullX = [x ; xPad] ;
y = vl_nntukeyloss(fullX, t) ;
end
