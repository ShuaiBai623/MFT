function cnn_benchmark(varargin)
%CNN_BENCHMARK Times execution of AutoNN and DagNN models

% Copyright (C) 2017 Joao F. Henriques.
% All rights reserved.
%
% This file is part of the VLFeat library and is made available under
% the terms of the BSD license (see the COPYING file).

  opts.models = {'imagenet-matconvnet-alex', ...
    'imagenet-googlenet-dag', 'imagenet-resnet-50-dag'} ;  % models to test (or absolute paths)
  opts.modelPath = [vl_rootnn() '/data/models'] ;  % path to find models given by name only
  opts.warmup = 3 ;  % number of warmup iterations (discarded)
  opts.trials = 5 ;  % number of benchmark iterations
  opts.gpu = [] ;  % GPU index, or empty for CPU mode
  opts.inputSizes = [] ;  % cell array with input size for each model, if net.meta contains no such info
  opts.batchSize = [] ;  % if non-empty, override batch size (4th dimension)
  opts.mode = 'normal' ;  % network evaluation mode ('normal' for backprop, or 'test')
  opts.dagnn = false ;  % if true, the networks are evaluated as DagNN objects instead of AutoNN
  opts.compileArgs = {} ;  % list of optional compilation arguments (see Net.compile)
  opts.modifier = @deal ;  % optional function to modify a network
  
  opts = vl_argparse(opts, varargin) ;
  
  if ischar(opts.models)  % single model
    opts.models = {opts.models} ;
  end
  
  if isempty(opts.inputSizes)  % no sizes specified
    opts.inputSizes = cell(size(opts.models)) ;
  end
  
  if ~isempty(opts.gpu)  % reset GPU
    gpu = gpuDevice(opts.gpu) ;
  end
  
  table = cell(numel(opts.models), 3) ;
  
  for m = 1:numel(opts.models)
    % get file name
    file = opts.models{m} ;
    [filedir, filename] = fileparts(file) ;
    if isempty(filedir)
      file = [opts.modelPath '/' file] ;
    end
    
    % load network
    net = load(file) ;
    
    % saved with "save <file> net", instead of "save <file> net -struct"
    if isstruct(net) && isfield(net, 'net')
      net = net.net ;
    end
    
    if opts.dagnn
      % load it as a DagNN, and keep it that way
      net = dagnn.DagNN.loadobj(net) ;
      net.mode = opts.mode ;
      assert(isempty(opts.compileArgs), ...
        'Compilation arguments not supported when benchmarking DagNN.') ;
    elseif isstruct(net)
      % constructor will convert any struct Net, DagNN or SimpleNN
      net = Net(net, opts.compileArgs{:}) ;
    else
      assert(isa(net, 'Net'), 'Expected a SimpleNN, DagNN or AutoNN model.') ;
    end
    
    % apply optional modifier function
    net = opts.modifier(net) ;
    
    % get input size, and name if present in meta information
    inputName = [] ;
    if ~isempty(opts.inputSizes{m})
      inputSize = opts.inputSizes{m} ;
    else
      assert(isstruct(net.meta), sprintf('Must specify input size for model %s.', filename)) ;
      if isfield(net.meta, 'inputSize')
        inputSize = net.meta.inputSize ;
      else
        try
          inputSize = net.meta.inputs(1).size ;
          inputName = net.meta.inputs(1).name ;
        catch
          error('Must specify input size for model %s.', filename) ;
        end
      end
    end
    
    % adjust input with a given batch size
    if ~isempty(opts.batchSize)
      inputSize(4) = opts.batchSize(min(m, end)) ;
    end
    
    % get input name
    if isempty(inputName)
      if ~opts.dagnn
        names = fieldnames(net.inputs) ;
      else
        names = net.getInputs() ;
      end
      assert(isscalar(names), 'Model must have only one input (image).')
      inputName = names{1} ;
    end
    
    % create appropriate output derivatives
    if ~opts.dagnn
      der = 1 ;
    else
      der = getDagNNDer(net, inputName, inputSize) ;
    end
    
    % create input, and move parameters/input/der to the GPU
    if isempty(opts.gpu)
      input = rand(inputSize, 'single') ;
    else
      reset(gpu) ;
      net.move('gpu') ;
      input = rand(inputSize, 'single', 'gpuArray') ;
      if ~opts.dagnn
        der = gpuArray(der) ;
      else
        for i = 2:2:numel(der)
          der{i} = gpuArray(der{i}) ;
        end
      end
      wait(gpu) ;
    end
    
    % run the network a few times
    time = zeros(opts.warmup + opts.trials, 1) ;
    for t = 1:numel(time)
      % run it
      tic() ;
      if ~opts.dagnn
        net.eval({inputName, input}, opts.mode, der) ;
      else
        net.eval({inputName, input}, der) ;
      end
      
      % wait for kernels to finish
      if ~isempty(opts.gpu)
        wait(gpu) ;
      end
      
      time(t) = toc() ;
    end
    
    % fill table with minimum and median time, skipping warm-up iterations
    time(1:opts.warmup) = [] ;
    table{m,1} = filename ;
    table{m,2} = num2str(min(time)) ;
    table{m,3} = num2str(median(time)) ;
  end
  
  % display table
  table = [{'Model', 'Min (s)', 'Median (s)'}; table] ;
  s = blanks(size(table,1))' ;
  disp([char(table{:,1}), s, s, char(table{:,2}), s, s, char(table{:,3})]) ;
end

function der = getDagNNDer(net, inputName, inputSize)
  % returns a list of correctly-sized output derivatives for a given DagNN
  sz = net.getVarSizes({inputName, inputSize}) ;
  outputNames = net.getOutputs() ;

  outputVars = net.getVarIndex(outputNames) ;
  
  der = cell(1, 2 * numel(outputNames)) ;
  der(1:2:end-1) = outputNames ;
  for i = 1:numel(outputNames)
    der{i * 2} = ones(sz{outputVars(i)}, 'single') ;
  end
end

