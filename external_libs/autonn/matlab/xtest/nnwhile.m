classdef nnwhile < nncheckder
  properties (TestParameter)
    conserveMemory = {false, true}
  end
  
  methods (Test)
    function testFor(test, conserveMemory)
      xi = Param('value', ones(1, test.currentDataType)) ;  % initial x (scalar)
      n = 4 ;  % number of iterations
      step = 1e-4 ;  % step for numerical derivative
      tol = 1e-3 ;  % tolerance for numerical check
      
      
      % single recursion variable
      iteration = @(x, t) 2 * x;
      
      xf = For(iteration, xi, n)  %#ok<*NOPRT>  % returns final x
      
      xf_ = test.checkDer(xf, conserveMemory, step, tol) ;  % returns the computed value of xf
      test.verifyEqual(xf_, xi.value * 2^n) ;
      
      
      % using counter
      iteration = @(x, t) x + t;
      
      xf = For(iteration, xi, n)
      
      xf_ = test.checkDer(xf, conserveMemory, step, tol) ;
      test.verifyEqual(xf_, xi.value + sum(1:n)) ;
      
      
      % non-recursive variable
      factor = Param('value', 3) ;
      iteration = @(x, t) factor * x;
      
      xf = For(iteration, xi, n)
      
      xf_ = test.checkDer(xf, conserveMemory, step, tol) ;
      test.verifyEqual(xf_, xi.value * factor.value^n) ;
      
      
      % two recursion variables
      yi = Param('value', zeros(1, test.currentDataType)) ;
      iteration = @(x, y, t) deal(2 * x, y + 1) ;
      
      [xf, yf] = For(iteration, xi, yi, n)
      
      xf_ = test.checkDer(xf, conserveMemory, step, tol) ;
      test.verifyEqual(xf_, xi.value * 2^n) ;
      
      yf_ = test.checkDer(yf, conserveMemory, step, tol) ;
      test.verifyEqual(yf_, yi.value + n) ;
      
      
      % concatenation of outputs over all iterations
      iteration = @(x, t) 2 * x;
      
      xf = For(iteration, xi, n, 'concatenate', 2)
      
      xf_ = test.checkDer(xf, conserveMemory, step, tol) ;
      test.verifyEqual(xf_, xi.value * 2 .^ (1:n)) ;
      
      
      % all together
      iteration = @(x, y, t) deal(factor * x, y + t) ;
      
      [xf, yf] = For(iteration, xi, yi, n, 'concatenate', 2)
      
      xf_ = test.checkDer(xf, conserveMemory, step, tol) ;
      test.verifyEqual(xf_, xi.value * factor.value .^ (1:n)) ;
      
      yf_ = test.checkDer(yf, conserveMemory, step, tol) ;
      test.verifyEqual(yf_, yi.value + cumsum(1:n)) ;
      
    end
  end
end

