% CUSTOM_DATASET_EXAMPLE
% Minimal demonstration of AutoNN training of a CNN on toy data.
% Data is loaded from memory, without reading any files.
%
% It also serves as a short tutorial on using custom data (as opposed to
% the standard datasets defined in 'autonn/+datasets/'). The toy data is
% generated on-the-fly (without reading from files). The same strategy
% could be used to return images that are stored in memory as an array,
% which is how the MNIST and CIFAR10 standard datasets work.
%
% While this strategy avoids file reading overhead, for large datasets this
% is not possible. See 'image_folder_example.m' for an example of creating
% a dataset by reading from a folder of images.
%
% The task is to distinguish between images of triangles, squares and
% circles.
%
% This example can be called with different name-value pairs, see the
% script below for a full list. Examples:
%
%  custom_dataset_example                         % use defaults
%  custom_dataset_example('learningRate', 0.005)  % a lower learning rate
%  custom_dataset_example('learningRate', 0.005, 'gpu', [])  % no GPU
%

function custom_dataset_example(varargin)
  % options (override by calling script with name-value pairs)
  opts.numEpochs = 10 ;  % number of epochs
  opts.batchSize = 200 ;  % batch size
  opts.learningRate = 0.01 ;  % learning rate
  opts.gpu = 1 ;  % GPU index, empty for CPU mode
  
  opts = vl_argparse(opts, varargin) ;  % let user override options
  
  try run('../../setup_autonn.m') ; catch; end  % add AutoNN to the path
  
  % set random seed
  rng('default') ;
  rng(0) ;
  

  % create network by composing layers/functions, starting with the inputs
  images = Input('gpu', true) ;  % automatically move images to the GPU if needed
  labels = Input() ;

  f = 1/100 ;
  x = vl_nnconv(images, 'size', [5, 5, 1, 5], 'weightScale', f) ;
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
  
  
  % show example images of the toy dataset (getSamples is defined below)
  figure(2) ;
  montage(getSamples(100), 'DisplayRange', [-0.5, 0.5]) ;
  title('Example images') ;


  % initialize solver
  solver = solvers.SGD('learningRate', opts.learningRate) ;
  
  % compute average objective and error
  stats = Stats({'objective', 'error'}) ;
  
  % enable GPU mode
  net.useGpu(opts.gpu) ;

  for epoch = 1:opts.numEpochs
    % training phase
    for batch = 1:25
      % draw samples
      [images, labels] = getSamples(opts.batchSize) ;

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
    for batch = 1:5
      [images, labels] = getSamples(opts.batchSize) ;

      net.eval({'images', images, 'labels', labels}, 'test') ;

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


% toy data generator: random images of triangles, squares and circles.
%
% note that you could replace this with any way to load and transform
% images or other inputs. avoid reading images from disk as it's slow.
%
% for efficient streaming from disk, it's best to use datasets.ImageFolder.

function [images, labels] = getSamples(batchSize)
  % randomly sample labels from 3 classes
  labels = randi(3, batchSize, 1) ;
  
  % number of polygon points in each class. the circle is created with 50
  % points.
  numPoints = [3, 4, 50] ;
  
  % allocate memory for a batch of images
  images = zeros(32, 32, 1, batchSize, 'single') ;
  
  % fill them in
  for i = 1:batchSize
    % points of a regular polygon, with random rotation and scale
    radius = randi([11, 14]) ;
    angles = rand(1) * 2 * pi + (0 : 2 * pi / numPoints(labels(i)) : 2 * pi) ;
    xs = 16.5 + cos(angles) * radius ;
    ys = 16.5 + sin(angles) * radius ;

    % generate image
    images(:,:,1,i) = poly2mask(xs, ys, 32, 32) ;
  end
  
  % approximately center the data (using the dataset mean would be better)
  images = images - 0.5 ;
end


