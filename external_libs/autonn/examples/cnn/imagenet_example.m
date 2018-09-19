% IMAGENET_EXAMPLE
% Demonstrates AutoNN training of a CNN on ImageNet.
% The task is image classification.
%
% This example can be called with different name-value pairs, see the
% script below for a full list. Examples:
%
%  imagenet_example                        % train AlexNet using defaults
%  imagenet_example('learningRate', 0.001) % override default learning rate
%  imagenet_example('model', models.ResNet())  % train a ResNet-50 model
%  imagenet_example('learningRate', 1e-4, 'solver', solvers.Adam()) % Adam solver
%  imagenet_example('dataDir', '/data/imagenet/')  % specify data location
%
%
% * ImageNet data setup:
%
% The ImageNet/ILSVRC data ships in several TAR archives that can be
% obtained from the ILSVRC challenge website. You will need:
% 
% ILSVRC2012_img_train.tar
% ILSVRC2012_img_val.tar
% ILSVRC2012_img_test.tar
% ILSVRC2012_devkit.tar
% 
% Note that images in the CLS-LOC challenge are the same for the 2012,
% 2013, and 2014 edition of ILSVRC, but that the development kit is
% different. However, all devkit versions should work.
%
% Create the following hierarchy:
%
% <dataDir>/images/train/ : content of ILSVRC2012_img_train.tar
% <dataDir>/images/val/ : content of ILSVRC2012_img_val.tar
% <dataDir>/images/test/ : content of ILSVRC2012_img_test.tar
% <dataDir>/ILSVRC2012_devkit : content of ILSVRC2012_devkit.tar
% 
% In order to speedup training and testing, it is a good idea to preprocess
% the images to have a fixed size (e.g. 256 pixels high). There is a script
% in MatConvNet (utils/preprocess-imagenet.sh) that achieves this. It may
% also be necessary to store the images in RAM disk. Reading images off
% disk with a sufficient speed is crucial for fast training.

function imagenet_example(varargin)
  % options (override by calling script with name-value pairs).
  % (*) if left empty, the default value for the chosen model will be used.
  opts.dataDir = [vl_rootnn() '/data/ilsvrc12'] ;  % ImageNet data location
  opts.resultsDir = [vl_rootnn() '/data/imagenet-example'] ;  % results location
  opts.model = models.AlexNet() ;  % choose model (type 'help models' for a list)
  opts.conserveMemory = true ;  % whether to conserve memory
  opts.numEpochs = [] ;  % epochs (*)
  opts.batchSize = [] ;  % batch size (*)
  opts.learningRate = [] ;  % learning rate (*)
  opts.weightDecay = [] ;  % weight decay (*)
  opts.solver = solvers.SGD() ;  % solver instance to use (type 'help solvers' for a list)
  opts.gpu = 1 ;  % GPU index, empty for CPU mode
  opts.numThreads = 12 ;  % number of threads for image reading
  opts.augmentation = [] ;  % data augmentation (see datasets.ImageFolder) (*)
  opts.savePlot = true ;  % whether to save the plot as a PDF file
  opts.continue = true ;  % continue from last checkpoint if available
  
  opts = vl_argparse(opts, varargin, 'nonrecursive') ;  % let user override options
  
  try run('../../setup_autonn.m') ; catch; end  % add AutoNN to the path
  mkdir(opts.resultsDir) ;
  

  % use chosen model's output as the predictions
  assert(isa(opts.model, 'Layer'), 'Model must be a CNN (e.g. models.AlexNet()).')
  predictions = opts.model ;
  
  % change the model's input name
  images = predictions.find('Input', 1) ;
  images.name = 'images' ;
  images.gpu = true ;
  
  % validate the prediction size (must predict 1000 classes)
  defaults = predictions.meta ;  % get model's meta information (default learning rate, etc)
  outputSize = predictions.evalOutputSize('images', [defaults.imageSize 5]) ;
  assert(isequal(outputSize, [1 1 1000 5]), 'Model output does not have the correct shape.') ;
  
  % replace empty options with model-specific default values
  for name_ = {'numEpochs', 'batchSize', 'learningRate', 'weightDecay', 'augmentation'}
    name = name_{1} ;  % dereference cell array
    if isempty(opts.(name))
      opts.(name) = defaults.(name) ;
    end
  end

  % create losses
  labels = Input() ;
  objective = vl_nnloss(predictions, labels, 'loss', 'softmaxlog') ;
  top1err = vl_nnloss(predictions, labels, 'loss', 'classerror') ;
  top5err = vl_nnloss(predictions, labels, 'loss', 'topkerror', 'topK', 5) ;

  % assign layer names automatically, and compile network
  Layer.workspaceNames() ;
  net = Net(objective, top1err, top5err, 'conserveMemory', opts.conserveMemory) ;


  % set solver learning rate
  solver = opts.solver ;
  solver.learningRate = opts.learningRate(1) ;
  solver.weightDecay = opts.weightDecay ;
  
  % initialize dataset
  dataset = datasets.ImageNet('dataDir', opts.dataDir, ...
    'imageSize', defaults.imageSize, 'useGpu', ~isempty(opts.gpu)) ;
  dataset.batchSize = opts.batchSize ;
  dataset.augmentation = opts.augmentation ;
  dataset.numThreads = opts.numThreads ;
  
  % compute average objective and error
  stats = Stats({'objective', 'top1err', 'top5err'}) ;
  
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
      
      % evaluate network to compute gradients
      tic;
      net.eval({'images', images, 'labels', labels}) ;
      
      % take one SGD step
      solver.step(net) ;

      % get current objective and error, and update their average.
      % also report iteration number and timing.
      t = toc() ;
      fprintf('ep%d %d/%d - %.1fms (%.1fHz) ', epoch, stats.counts(1) + 1, ...
        floor(numel(dataset.trainSet) / opts.batchSize), t * 1e3, opts.batchSize / t);
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
      t = toc() ;

      fprintf('val ep%d %d/%d - %.1fms (%.1fHz) ', epoch, stats.counts(1) + 1, ...
        floor(numel(dataset.valSet) / opts.batchSize), t * 1e3, opts.batchSize / t);
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

