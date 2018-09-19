classdef nntrain < nntest
  properties (Constant)
    % test these solvers on MNIST
    solvers = {@solvers.SGD, @solvers.Adam, @solvers.AdaGrad, @solvers.AdaDelta, @solvers.RMSProp}
    trainObjective = [ 0.29,          0.14,             0.16,              0.40,             0.16]
    valObjective = [  0.078,         0.095,            0.065,             0.125,            0.060]
  end
  
  properties (TestParameter)
    solverIdx = num2cell(1:numel(nntrain.solvers))  % iterate solvers
  end
  
  methods (TestClassSetup)
    function init(test)  %#ok<MANU>
      % place mnist_example on path
      addpath([fileparts(mfilename('fullpath')) '/../../examples/cnn']);
    end
  end

  methods (Test)
    function testObjective(test, solverIdx)
      if strcmp(test.currentDataType, 'double')
        return
      end
      
      rng(0);  % fix random seed, for reproducible tests
      
      switch test.currentDevice
      case 'cpu', gpu = [];
      case 'gpu', gpu = 1;
      end
      
      % instantiate solver
      solver = nntrain.solvers{solverIdx}() ;
      
      % run training
      [~, stats] = mnist_example('solver', solver, 'numEpochs', 1, ...
        'gpu', gpu, 'resultsDir', []) ;
      
      % check objective
      value = stats.values('train', 'objective') ;
      target = nntrain.trainObjective(solverIdx) ;
      test.verifyLessThan(value, target * 1.1);  % tolerance of 10%
      
      value = stats.values('val', 'objective') ;
      target = nntrain.valObjective(solverIdx) ;
      test.verifyLessThan(value, target * 1.1);  % tolerance of 10%
    end
  end
end
