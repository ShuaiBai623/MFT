% CIFAR_EXAMPLE
% Demonstrates AutoNN training of a CNN on CIFAR10.
% The task is small-image classification (10 classes).
%
% This example can be called with different name-value pairs, see the
% script below for a full list. Examples:
%
%  cifar_example                           % train a fast, basic network
%  cifar_example('learningRate', 0.001)    % override default learning rate
%  cifar_example('learningRate', 1e-4, 'solver', solvers.Adam()) % Adam solver
%  cifar_example('model', models.MaxoutNIN())  % Maxout-NIN model (~8% error, fast)
%  cifar_example('model', models.ResNet('numClasses', 10))  % ResNet-50 model
%  cifar_example('dataDir', '~/cifar')   % dataset path (downloaded automatically)
%  cifar_example('resultsDir', '~/out', 'savePlot', true)   % plot to ~/out
%

function cifar_example(varargin)
  % options (override by calling script with name-value pairs)
  opts.dataDir = [vl_rootnn() '/data/cifar'] ;  % CIFAR10 data location
  opts.resultsDir = [vl_rootnn() '/data/cifar-example'] ;  % results location
  opts.model = models.BasicCifarNet() ;  % choose model (type 'help models' for a list)
  opts.conserveMemory = true ;  % whether to conserve memory (helpful with e.g. @models.MaxoutNIN)
  opts.numEpochs = [] ;  % if empty, default for above model will be used
  opts.batchSize = [] ;  % same as above
  opts.learningRate = [] ;  % same as above
  opts.weightDecay = [] ;  % same as above
  opts.solver = solvers.SGD() ;  % solver instance to use (type 'help solvers' for a list)
  opts.gpu = 1 ;  % GPU index, empty for CPU mode
  opts.savePlot = true ;  % whether to save the plot as a PDF file
  opts.continue = false ;  % continue from last checkpoint if available
  
  opts = vl_argparse(opts, varargin, 'nonrecursive') ;  % let user override options
  
  try run('../../setup_autonn.m') ; catch; end  % add AutoNN to the path
  mkdir(opts.resultsDir) ;
  

  % use chosen model's output as the predictions
  assert(isa(opts.model, 'Layer'), 'Model must be a CNN (e.g. models.NIN()).')
  predictions = opts.model ;
  
  % change the model's input name
  images = predictions.find('Input', 1) ;
  images.name = 'images' ;
  images.gpu = true ;
  
  % validate the prediction size (must predict 10 classes)
  sz = predictions.evalOutputSize('images', [32 32 3 5]) ;
  assert(isequal(sz, [1 1 10 5]), 'Model output does not have the correct shape.') ;
  
  % replace empty options with the model-specific default values
  defaults = predictions.meta ;  % get model's meta information (default learning rate, etc)
  if isempty(opts.numEpochs),    opts.numEpochs = defaults.numEpochs ; end
  if isempty(opts.batchSize),    opts.batchSize = defaults.batchSize ; end
  if isempty(opts.learningRate), opts.learningRate = defaults.learningRate ; end
  if isempty(opts.weightDecay),  opts.weightDecay = defaults.weightDecay ; end

  % create losses
  labels = Input() ;
  objective = vl_nnloss(predictions, labels, 'loss', 'softmaxlog') ;
  error = vl_nnloss(predictions, labels, 'loss', 'classerror') ;

  % assign layer names automatically, and compile network
  Layer.workspaceNames() ;
  net = Net(objective, error, 'conserveMemory', opts.conserveMemory) ;


  % set solver learning rate
  solver = opts.solver ;
  solver.learningRate = opts.learningRate(1) ;
  solver.weightDecay = opts.weightDecay ;
  
  % initialize dataset
  dataset = datasets.CIFAR10(opts.dataDir, 'batchSize', opts.batchSize) ;
  
  % compute average objective and error
  stats = Stats({'objective', 'error'}) ;
  
  % continue from last checkpoint if there is one
  startEpoch = 1 ;
  if opts.continue
    [filename, startEpoch] = models.checkpoint([opts.resultsDir '/epoch-*.mat']) ;
  end
  if startEpoch > 1
    load(filename, 'net', 'stats', 'solver') ;
  end

  % enable GPU mode
  net.useGpu(opts.gpu) ;

  for epoch = startEpoch : opts.numEpochs
    % get the learning rate for this epoch, if there is a schedule
    if epoch <= numel(opts.learningRate)
      solver.learningRate = opts.learningRate(epoch) ;
    end
    
    % training phase
    for batch = dataset.train()
      % draw samples
      [images, labels] = dataset.get(batch) ;
      
      % simple data augmentation: flip images horizontally
      if rand() > 0.5, images = fliplr(images) ; end
      
      % evaluate network to compute gradients
      tic;
      net.eval({'images', images, 'labels', labels}) ;
      
      % take one SGD step
      solver.step(net) ;

      % get current objective and error, and update their average.
      % also report iteration number and timing.
      fprintf('ep%d %d - %.1fms ', epoch, stats.counts(1) + 1, toc() * 1000);
      stats.update(net) ;
      stats.print() ;
    end
    % push average objective and error (after one epoch) into the plot
    stats.push('train') ;

    % validation phase
    for batch = dataset.val()
      [images, labels] = dataset.get(batch) ;

      tic;
      net.eval({'images', images, 'labels', labels}, 'test') ;

      fprintf('val ep%d %d - %.1fms ', epoch, stats.counts(1) + 1, toc() * 1000);
      stats.update(net) ;
      stats.print() ;
    end
    stats.push('val') ;

    % plot statistics
    stats.plot('figure', 1) ;

    if ~isempty(opts.resultsDir)
      % save the plot
      if opts.savePlot
        print(1, [opts.resultsDir '/plot.pdf'], '-dpdf') ;
      end

      % save checkpoint every few epochs
      if mod(epoch, 10) == 0
        save(sprintf('%s/epoch-%d.mat', opts.resultsDir, epoch), ...
          'net', 'stats', 'solver') ;
      end
    end
  end

  % save results
  if ~isempty(opts.resultsDir)
    save([opts.resultsDir '/results.mat'], 'net', 'stats', 'solver') ;
  end
end

