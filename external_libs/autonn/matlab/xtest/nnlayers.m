classdef nnlayers < nncheckder
  properties (TestParameter)
    conserveMemory = {false, true}
  end
  properties
    currentConserveMemory
  end

  methods (Test)
    function testLayers(test, conserveMemory)
      test.currentConserveMemory = conserveMemory ;
      
      % use Params for all inputs so we can choose their values now
      x = Param('value', randn(7, 7, 2, 5, test.currentDataType)) ;
      w = Param('value', randn(3, 3, 2, 3, test.currentDataType)) ;
      b = Param('value', randn(3, 1, test.currentDataType)) ;
      labels = Param('value', ones(5, 1, test.currentDataType)) ;
      Layer.workspaceNames() ;
      
      % test several layers and syntaxes
      
      do(test, vl_nnrelu(x)) ;
      
      do(test, vl_nnconv(x, w, b)) ;
      
      do(test, vl_nnconv(x, w, b, 'stride', 3, 'pad', 2)) ;
      
      do(test, vl_nnconvt(x, permute(w, [1 2 4 3]), b)) ;
      
      do(test, vl_nnconvt(x, permute(w, [1 2 4 3]), b, 'upsample', 3, 'crop', 2)) ;
      
      do(test, vl_nnpool(x, 2)) ;
      
      do(test, vl_nnpool(x, [2, 2], 'stride', 2, 'pad', 1)) ;
      
      do(test, vl_nnloss(x, labels, 'loss', 'classerror'), labels) ;
      
      % dropout is composed of 2 parts: the mask generator, and the dropout
      % mask applier. run derivative check with fixed mask.
      rate = 0.1 ;
      dropout = vl_nndropout(x, 'rate', rate) ;
      mask = dropout.inputs{2} ;
      
      test.verifyInstanceOf(mask, 'Layer') ;
      test.verifyEqual(mask.func, @vl_nnmask) ;
      
      dropout.inputs{2} = vl_nnmask(x.value, rate) ;  % make it constant
      do(test, dropout) ;
      
      % bnorm params are single
      if strcmp(test.currentDataType, 'single')
        % batch-norm needs special handling
        bnorm = vl_nnbnorm(x) ;
        
        % replicate gain/bias for all output channels (ordinarily done by
        % the solver on first update)
        bnorm.inputs{2}.value = bnorm.inputs{2}.value([1; 1]) ;  % gain
        bnorm.inputs{3}.value = bnorm.inputs{3}.value([1; 1]) ;  % bias
        
        % ignore the 4th parameter (moments), since it is not updated by
        % gradient descent but by a moving average
        ignore = bnorm.inputs{4} ;
        
        do(test, bnorm, ignore) ;
      end
    end
    
    function testMath(test, conserveMemory)
      test.currentConserveMemory = conserveMemory ;
      
      % use Params for all inputs so we can choose their values now
      a = Param('value', randn(3, 3, test.currentDataType) + 0.1 * eye(3,3)) ;  % matrix
      b = Param('value', randn(3, 3, test.currentDataType) + 0.1 * eye(3,3)) ;  % matrix
      c = Param('value', 2 * ones(1, 1, test.currentDataType)) ;  % scalar
      d = Param('value', randn(3, 1, test.currentDataType)) ;  % vector
      e = Param('value', rand(3, 3, test.currentDataType) + 1e-3 * ones(3,3)) ;  % non-negative matrix
      f = Param('value', rand(3, 3, test.currentDataType) * 2 - 1) ;  % matrix in -1..1
      Layer.workspaceNames() ;
      
      % test several operations
      
      % weighted sums
      do(test, a + b) ;
      do(test, 10 * a) ;
      do(test, a + 2 * b - c) ;  % collected arguments in a single wsum
      
      % matrix
      do(test, a * b) ;
      do(test, a') ;
      do(test, inv(a)) ;
      do(test, a / b) ;
      do(test, a \ b) ;
      
      % binary with expanded dimensions
      do(test, a .* d) ;
      do(test, a ./ d) ;
      do(test, a .^ c, [], 1e-6 * test.range, 1e-3) ;  % higher tolerance
      
      do(test, atan2(a, b)) ;
      
      % matrix ops should deal with scalars gracefully
      do(test, a * c) ;
      do(test, a / c) ;
      
      % unary
      do(test, sqrt(e)) ;
      do(test, sin(a)) ;
      do(test, cos(a)) ;
      do(test, tan(a)) ;
      do(test, asin(f)) ;
      do(test, acos(f)) ;
      do(test, atan(a)) ;

      % sorting is a kind of math
      do(test, sort(a)) ;
    end
    
    function testConv(test, conserveMemory)
      test.currentConserveMemory = conserveMemory ;
      
      % extra conv tests
      if strcmp(test.currentDataType, 'double'), return, end
      
      x = Param('value', randn(7, 7, 2, 5, test.currentDataType)) ;
      
      % 'size' syntax
      do(test, vl_nnconv(x, 'size', [7, 7, 2, 5])) ;
      
      % bias
      layer = vl_nnconv(x, 'size', [7, 7, 2, 5], 'hasBias', false) ;
      do(test, layer) ;
      test.verifyEmpty(layer.inputs{3}) ;
      
      % Param learning arguments
      layer = vl_nnconv(x, 'size', [7, 7, 2, 5], ...
          'learningRate', [1, 2], 'weightDecay', [3, 4]) ;
      do(test, layer) ;
      test.eq(layer.inputs{2}.learningRate, 1) ;
      test.eq(layer.inputs{3}.learningRate, 2) ;
      test.eq(layer.inputs{2}.weightDecay, 3) ;
      test.eq(layer.inputs{3}.weightDecay, 4) ;
    end
  end
  
  methods
    function do(test, output, varargin)
      % show layer for debugging
      display(output) ;
      
      % use parent class's derivative check (defined in nncheckder)
      test.checkDer(output, test.currentConserveMemory, varargin{:}) ;
    end
  end
end
