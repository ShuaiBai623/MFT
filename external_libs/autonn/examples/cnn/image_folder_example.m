% IMAGE_FOLDER_EXAMPLE
% Minimal demonstration of AutoNN training of a CNN on toy data.
% Data is loaded from a folder with JPEG images.
%
% It also serves as a short tutorial on using custom data (as opposed to
% the standard datasets defined in 'autonn/+datasets/'). The toy data is
% read by using the datasets.ImageFolder class.
%
% While the image reading is multi-threaded to be as fast as possible, it
% is even faster to simply keep images in memory and avoid reading files.
% See 'custom_dataset_example.m' for an example that avoids image reading.
% That is only feasible for small datasets (such as MNIST or CIFAR).
%
% The task is to distinguish between images of triangles, squares and
% circles.
%
% This example can be called with different name-value pairs, see the
% script below for a full list. Examples:
%
%  image_folder_example                         % use defaults
%  image_folder_example('learningRate', 0.005)  % a lower learning rate
%  image_folder_example('learningRate', 0.005, 'gpu', [])  % no GPU
%

function image_folder_example(varargin)
  % options (override by calling script with name-value pairs)
  opts.dataDir = [vl_rootnn() '/data/shapes'] ;  % data location (created automatically)
  opts.numEpochs = 10 ;  % number of epochs
  opts.batchSize = 200 ;  % batch size
  opts.learningRate = 0.01 ;  % learning rate
  opts.gpu = 1 ;  % GPU index, empty for CPU mode
  
  opts = vl_argparse(opts, varargin) ;  % let user override options
  
  try run('../../setup_autonn.m') ; catch; end  % add AutoNN to the path
  
  
  % generate the toy data if it does not exist
  if ~exist([opts.dataDir '/train'], 'dir')
    generateToyData(opts.dataDir) ;
  end
  
  
  % set random seed
  rng('default') ;
  rng(0) ;
  

  % create network by composing layers/functions, starting with the inputs
  images = Input('gpu', true) ;  % automatically move images to the GPU if needed
  labels = Input() ;

  f = 1/100 ;
  x = vl_nnconv(images, 'size', [5, 5, 3, 5], 'weightScale', f) ;
  x = vl_nnpool(x, 2, 'stride', 2) ;
  x = vl_nnconv(x, 'size', [5, 5, 5, 10], 'weightScale', f) ;
  x = vl_nnpool(x, 2, 'stride', 2) ;
  x = vl_nnconv(x, 'size', [5, 5, 10, 3], 'weightScale', f) ;

  objective = vl_nnloss(x, labels) ;  % what we minimize
  error = vl_nnloss(x, labels, 'loss', 'classerror') ;  % the error metric

  % assign layer names based on workspace variables ('images', 'objective')
  Layer.workspaceNames() ;
  
  % compile network
  net = Net(objective, error) ;
  
  
  % initialize dataset. the data directory will be automatically scanned
  % for JPEG files.
  dataset = datasets.ImageFolder('dataDir', opts.dataDir, ...
    'imageSize', [32, 32], 'batchSize', opts.batchSize) ;
  
  % iterate images, and guess their sets/labels from directory structure
  n = numel(dataset.filenames) ;
  labels = zeros(n, 1) ;
  isVal = false(n, 1) ;
  for i = 1:n
    % the format is: '<set>/<label>/<sample>.jpg'
    file = dataset.filenames{i} ;
    parts = strsplit(file, '/') ;
    if strcmp(parts{1}, 'val')
      isVal(i) = true ;
    end
    labels(i) = str2double(parts{2}) ;  % numeric label
  end
  
  % assign train/val sets
  dataset.trainSet = find(~isVal) ;
  dataset.valSet = find(isVal) ;
  
  % show example images of the toy dataset
  figure(2) ;
  montage(dataset.get(randperm(n, 100)), 'DisplayRange', [-0.5, 0.5]) ;
  title('Example images') ;


  % initialize solver
  solver = solvers.SGD('learningRate', opts.learningRate) ;
  
  % compute average objective and error
  stats = Stats({'objective', 'error'}) ;
  
  % enable GPU mode
  net.useGpu(opts.gpu) ;

  for epoch = 1:opts.numEpochs
    % training phase
    for batch = dataset.train()  % batch will be wrapped in a scalar cell array
      % draw samples
      images = dataset.get(batch) ;

      % evaluate network to compute gradients
      batch = batch{1} ;  % remove the scalar cell array
      net.eval({'images', images, 'labels', labels(batch)}) ;
      
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
      images = dataset.get(batch) ;

      batch = batch{1} ;  %#ok<*FXSET>
      net.eval({'images', images, 'labels', labels(batch)}, 'test') ;

      stats.update(net) ;
      stats.print() ;
    end
    stats.push('val') ;
    
    
    % debug: visualize the learned filters
    figure(3) ; vl_tshow(net.getValue('conv1_filters')) ; title('Conv1 filters') ;
    figure(4) ; vl_tshow(net.getValue('conv2_filters')) ; title('Conv2 filters') ;
    figure(5) ; vl_tshow(net.getValue('x_filters')) ; title('Conv3 filters') ;

    % plot statistics
    stats.plot('figure', 1) ;
  end
end


function generateToyData(dataDir)
% Generates toy data in the given path: random image of triangles, squares
% and circles.
%
% The directory format is: '<dataDir>/<set>/<label>/<sample>.jpg', where
% <set> is 'train' or 'val', <label> is an integer between 1 and 3, and
% <sample> is the sample index.

  % Set random seed
  rng('default') ;
  rng(0) ;

  % The sets, and number of samples per label in each set
  sets = {'train', 'val'} ;
  numSamples = [1500, 150] ;

  % Number of polygon points in each class. The circle is created with 50
  % points.
  numPoints = [3, 4, 50] ;
  
  for s = 1:2  % Iterate sets
    for label = 1:3  % Iterate labels
      fprintf('Generating images for set %s, label %i...\n', sets{s}, label) ;
      
      mkdir(sprintf('%s/%s/%i', dataDir, sets{s}, label)) ;
      
      for i = 1:numSamples(s)  % Iterate samples
        % Points of a regular polygon, with random rotation and scale
        radius = randi([11, 14]) ;
        angles = rand(1) * 2 * pi + (0 : 2 * pi / numPoints(label) : 2 * pi) ;
        xs = 16.5 + cos(angles) * radius ;
        ys = 16.5 + sin(angles) * radius ;

        % Generate image
        image = poly2mask(xs, ys, 32, 32) ;
        
        % Save it
        imwrite(image, sprintf('%s/%s/%i/%04i.jpg', dataDir, sets{s}, label, i)) ;
      end
    end
  end

end

