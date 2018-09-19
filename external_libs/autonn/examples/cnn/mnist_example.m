% MNIST_EXAMPLE
% Minimal demonstration of AutoNN training of a CNN on MNIST.
% The task is handwritten digit recognition.
%
% This example can be called with different name-value pairs, see the
% script below for a full list. Examples:
%
%  mnist_example                          % use defaults
%  mnist_example('learningRate', 0.0005)  % a lower learning rate
%  mnist_example('learningRate', 0.0005, 'gpu', [])  % no GPU
%  mnist_example('dataDir', '~/mnist')  % dataset path (downloaded automatically)
%  mnist_example('resultsDir', '~/out', 'savePlot', true)  % plot to ~/out
%

function [net, stats] = mnist_example(varargin)
  % options (override by calling script with name-value pairs)
  opts.dataDir = [vl_rootnn() '/data/mnist'] ;  % MNIST data location
  opts.resultsDir = [vl_rootnn() '/data/mnist-example'] ;  % results location
  opts.numEpochs = 20 ;  % number of epochs
  opts.batchSize = 128 ;  % batch size
  opts.learningRate = 0.001 ;  % learning rate
  opts.solver = solvers.SGD() ;  % solver instance to use (type 'help solvers' for a list)
  opts.gpu = 1 ;  % GPU index, empty for CPU mode
  opts.savePlot = false ;  % whether to save the plot as a PDF file
  
  opts = vl_argparse(opts, varargin) ;  % let user override options
  
  try run('../../setup_autonn.m') ; catch; end  % add AutoNN to the path
  mkdir(opts.resultsDir) ;
  

  % build network. we could also have used:
  %   output = models.LeNet();
  % or any other model from 'autonn/matlab/+models/'.
  
  % create network inputs
  images = Input('gpu', true) ;
  labels = Input() ;
  
  x = vl_nnconv(images, 'size', [5, 5, 1, 20], 'weightScale', 0.01) ;
  x = vl_nnpool(x, 2, 'stride', 2) ;
  
  x = vl_nnconv(x, 'size', [5, 5, 20, 50], 'weightScale', 0.01) ;
  x = vl_nnpool(x, 2, 'stride', 2) ;
  
  x = vl_nnconv(x, 'size', [4, 4, 50, 500], 'weightScale', 0.01) ;
  x = vl_nnrelu(x) ;
  
  output = vl_nnconv(x, 'size', [1, 1, 500, 10], 'weightScale', 0.01) ;

  % create losses
  objective = vl_nnloss(output, labels, 'loss', 'softmaxlog') ;
  error = vl_nnloss(output, labels, 'loss', 'classerror') ;

  % assign layer names automatically, and compile network
  Layer.workspaceNames() ;
  net = Net(objective, error) ;


  % set solver learning rate
  solver = opts.solver ;
  solver.learningRate = opts.learningRate ;
  
  % initialize dataset
  dataset = datasets.MNIST(opts.dataDir, 'batchSize', opts.batchSize) ;
  
  % compute average objective and error
  stats = Stats({'objective', 'error'}) ;
  
  % enable GPU mode
  net.useGpu(opts.gpu) ;

  for epoch = 1:opts.numEpochs
    % training phase
    for batch = dataset.train()
      % draw samples
      [images, labels] = dataset.get(batch) ;

      % evaluate network to compute gradients
      net.eval({'images', images, 'labels', labels}) ;
      
      % take one SGD step
      solver.step(net) ;

      % get current objective and error, and update their average
      stats.update(net) ;
      stats.print() ;
    end
    % push average objective and error (after one epoch) into the plot
    stats.push('train') ;

    % validation phase
    for batch = dataset.val()
      [images, labels] = dataset.get(batch) ;

      net.eval({'images', images, 'labels', labels}, 'test') ;

      stats.update(net) ;
      stats.print() ;
    end
    stats.push('val') ;

    % plot statistics
    stats.plot('figure', 1) ;
    if opts.savePlot && ~isempty(opts.resultsDir)
      print(1, [opts.resultsDir '/plot.pdf'], '-dpdf') ;
    end
  end

  % save results
  if ~isempty(opts.resultsDir)
    save([opts.resultsDir '/results.mat'], 'net', 'stats', 'solver') ;
  end
end

