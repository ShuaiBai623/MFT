function [net,stats] = cnn_train_autonn(net, imdb, getBatch, varargin)
%CNN_TRAIN_AUTONN Demonstrates training a CNN using the AutoNN wrapper
%   CNN_TRAIN_AUTONN is similar to CNN_TRAIN, but works with the AutoNN
%   wrapper instead of the SimpleNN wrapper.

% Copyright (C) 2014-18 Andrea Vedaldi and Joao F. Henriques.
% All rights reserved.
%
% This file is part of the VLFeat library and is made available under
% the terms of the BSD license (see the COPYING file).
addpath(fullfile(vl_rootnn, 'examples'));


warning('AutoNN:cnn_train_autonn', ['The function cnn_train_autonn is ' ...
  'deprecated. It will still be supported, however switching to the new, ' ...
  'more flexible training classes (Solver, Dataset, Stats) is recommended. ' ...
  'See the examples for more information. This warning will now be disabled.']) ;
warning('off', 'AutoNN:cnn_train_autonn') ;


% this is needed to harmonize the behavior of two versions of vl_nnloss:
% the legacy behavior which *sums* the loss over the batch, and the new
% behavior that takes the *average* over the batch.
% first, detect if the new behavior ('normalise' option) is present.
old = false ;
try
  vl_nnloss([], [], 'normalise', true)  ;
catch  % unrecognized option, must be the old vl_nnloss
  old = true ;
end
if old
  warning('AutoNN:normalizedLoss', ['The most recent version of ' ...
   'vl_nnloss normalizes the loss by the batch size. The current version ' ...
   'does not. A workaround is being used, but consider updating MatConvNet.']) ;
end
% consider the loss normalized (workaround in Layer.vl_nnloss). this can
% be disabled if detrimental (e.g. custom loss without normalization).
opts.normalizedLoss = true ;


opts.expDir = fullfile('data','exp') ;
opts.continue = true ;
opts.batchSize = 256 ;
opts.numSubBatches = 1 ;
opts.train = [] ;
opts.val = [] ;
opts.gpus = [] ;
opts.prefetch = false ;
opts.numEpochs = 300 ;
opts.learningRate = 0.001 ;
opts.weightDecay = 0.0005 ;

opts.solver = [] ;  % Empty array means use the default SGD solver
[opts, varargin] = vl_argparse(opts, varargin) ;
if ~isempty(opts.solver)
  assert(isa(opts.solver, 'function_handle') && nargout(opts.solver) == 2,...
    'Invalid solver; expected a function handle with two outputs.') ;
  % Call without input arguments, to get default options
  opts.solverOpts = opts.solver() ;
end

opts.onNanInf = [] ;  % Behavior when variables with NaN/Inf are found: 'error', 'warning' or 'debug' (default: nothing).

opts.momentum = 0.9 ;
opts.saveSolverState = true ;
opts.nesterovUpdate = false ;
opts.randomSeed = 0 ;
opts.profile = false ;
opts.parameterServer.method = 'mmap' ;
opts.parameterServer.prefix = 'mcn' ;

if isa(net, 'dagnn.DagNN')
  opts.derOutputs = {'objective', 1} ;
else
  opts.derOutputs = 1 ;
end
opts.stats = 'auto' ;  % list of layers to aggregate stats from (e.g. loss, error), or 'auto' for automatic
opts.extractStatsFn = [] ;
opts.plotStatistics = true;
opts.plotDiagnostics = false ;
opts.postEpochFn = [] ;  % postEpochFn(net,params,state) called after each epoch; can return a new learning rate, 0 to stop, [] for no change
opts = vl_argparse(opts, varargin) ;

if ~exist(opts.expDir, 'dir'), mkdir(opts.expDir) ; end
if isempty(opts.train), opts.train = find(imdb.images.set==1) ; end
if isempty(opts.val), opts.val = find(imdb.images.set==2) ; end
if isscalar(opts.train) && isnumeric(opts.train) && isnan(opts.train)
  opts.train = [] ;
end
if isscalar(opts.val) && isnumeric(opts.val) && isnan(opts.val)
  opts.val = [] ;
end

% -------------------------------------------------------------------------
%                                                            Initialization
% -------------------------------------------------------------------------

evaluateMode = isempty(opts.train) ;
if ~evaluateMode
  if isempty(opts.derOutputs)
    error('DEROUTPUTS must be specified when training.\n') ;
  end
end

% find loss layers now instead of at every extractStatsFn call
if isa(net, 'dagnn.DagNN')
  if isempty(opts.extractStatsFn)
    opts.extractStatsFn = @extractStats ;
  end
  
  if isequal(opts.stats, 'auto')  % all losses
    sel = find(cellfun(@(x) isa(x,'dagnn.Loss'), {net.layers.block})) ;
  else
    sel = zeros(size(opts.stats)) ;
    for i = 1:numel(sel)  % find layers with the given names
      sel(i) = net.getLayerIndex(opts.stats{i}) ;
    end
  end
else  % AutoNN
  if isempty(opts.extractStatsFn)
    opts.extractStatsFn = @extractStatsAutoNN ;
  end
  
  if isequal(opts.stats, 'auto')  % all losses
    sel = find(cellfun(@(f) isequal(f, @vl_nnloss) || ...
      isequal(f, @vl_nnsoftmaxloss), {net.forward.func})) ;
  else
    sel = zeros(size(opts.stats)) ;
    for i = 1:numel(sel)  % find layers with the given names
      sel(i) = find(strcmp({net.forward.name}, opts.stats{i})) ;
    end
  end
end
fn = opts.extractStatsFn ;
normalizedLoss = opts.normalizedLoss ;
opts.extractStatsFn = @(stats, net, batchSize) fn(stats, net, sel, batchSize, normalizedLoss) ;

% -------------------------------------------------------------------------
%                                                        Train and validate
% -------------------------------------------------------------------------

modelPath = @(ep) fullfile(opts.expDir, sprintf('net-epoch-%d.mat', ep));
modelFigPath = fullfile(opts.expDir, 'net-train.pdf') ;

start = opts.continue * findLastCheckpoint(opts.expDir) ;
if start >= 1
  fprintf('%s: resuming by loading epoch %d\n', mfilename, start) ;
  [net, state, stats] = loadState(modelPath(start)) ;
else
  state = [] ;
end

for epoch=start+1:opts.numEpochs

  % Set the random seed based on the epoch and opts.randomSeed.
  % This is important for reproducibility, including when training
  % is restarted from a checkpoint.

  rng(epoch + opts.randomSeed) ;
  prepareGPUs(opts, epoch == start+1) ;

  % Train for one epoch.
  params = opts ;
  params.epoch = epoch ;
  params.learningRate = opts.learningRate(min(epoch, numel(opts.learningRate))) ;
  params.train = opts.train(randperm(numel(opts.train))) ; % shuffle
  params.val = opts.val(randperm(numel(opts.val))) ;
  params.imdb = imdb ;
  params.getBatch = getBatch ;

  if numel(opts.gpus) <= 1
    [net, state] = processEpoch(net, state, params, 'train') ;
    [net, state] = processEpoch(net, state, params, 'val') ;
    if ~evaluateMode
      saveState(modelPath(epoch), net, state) ;
    end
    lastStats = state.stats ;
  else
    spmd
      [net, state] = processEpoch(net, state, params, 'train') ;
      [net, state] = processEpoch(net, state, params, 'val') ;
      if labindex == 1 && ~evaluateMode
        saveState(modelPath(epoch), net, state) ;
      end
      lastStats = state.stats ;
    end
    lastStats = accumulateStats(lastStats) ;
  end
  
  if ~isempty(opts.postEpochFn)
    state.stats = lastStats ;  % allow the func. to read accumulated stats
    if nargout(opts.postEpochFn) <= 0
      opts.postEpochFn(net, params, state) ;
    else  % returned an updated learning rate
      if nargout(opts.postEpochFn) == 1
        lr = opts.postEpochFn(net, params, state) ;
      else  % returned updated/custom stats 
        [lr, lastStats] = opts.postEpochFn(net, params, state) ;
      end
      if ~isempty(lr), opts.learningRate = lr; end
      if opts.learningRate == 0, break; end
    end
  end

  stats.train(epoch) = lastStats.train ;
  stats.val(epoch) = lastStats.val ;
  clear lastStats ;
  saveStats(modelPath(epoch), stats) ;

  if opts.plotStatistics
    switchFigure(1) ; clf ;
    plots = setdiff(...
      cat(2,...
      fieldnames(stats.train)', ...
      fieldnames(stats.val)'), {'num', 'time'}) ;
    for p = plots
      p = char(p) ;
      values = zeros(0, epoch) ;
      leg = {} ;
      for f = {'train', 'val'}
        f = char(f) ;
        if isfield(stats.(f), p)
          tmp = [stats.(f).(p)] ;
          values(end+1,:) = tmp(1,:)' ;
          leg{end+1} = f ;
        end
      end
      subplot(1,numel(plots),find(strcmp(p,plots))) ;
      plot(1:epoch, values','o-') ;
      xlabel('epoch') ;
      title(p) ;
      legend(leg{:}) ;
      grid on ;
    end
    drawnow ;
    print(1, modelFigPath, '-dpdf') ;
  end
end

% With multiple GPUs, return one copy
if isa(net, 'Composite'), net = net{1} ; end

% -------------------------------------------------------------------------
function [net, state] = processEpoch(net, state, params, mode)
% -------------------------------------------------------------------------
% Note that net is not strictly needed as an output argument as net
% is a handle class. However, this fixes some aliasing issue in the
% spmd caller.

% initialize with momentum 0
if isempty(state) || isempty(state.solverState)
  state.solverState = cell(1, numel(net.params)) ;
  state.solverState(:) = {0} ;
end

% move CNN  to GPU as needed
numGpus = numel(params.gpus) ;
if numGpus >= 1
  net.move('gpu') ;
  for i = 1:numel(state.solverState)
    if ~isa(net, 'dagnn.DagNN') && ~net.isGpuVar(net.params(i).var)
      continue
    end
    s = state.solverState{i} ;
    if isnumeric(s)
      state.solverState{i} = gpuArray(s) ;
    elseif isstruct(s)
      state.solverState{i} = structfun(@gpuArray, s, 'UniformOutput', false) ;
    end
  end
end
if numGpus > 1
  parserv = ParameterServer(params.parameterServer) ;
  
  if ~isa(net, 'dagnn.DagNN')
    % some layers have implicit Param sizes: the Params are scalar, until
    % the first derivative computation, when they're expanded to full size.
    % this allows initializing layers without knowing their input/output
    % sizes, making them less verbose. e.g. batch-norm: y = vl_nnbnorm(x);
    % for multi-GPU we need their sizes, so compute their derivatives once.
    subset = params.(mode) ;
    inputs = params.getBatch(params.imdb, subset(1)) ;  % one sample
    
    net.eval(inputs, 'normal', params.derOutputs) ;
    
    paramDer = net.getDer([net.params.var]) ;
    if ~iscell(paramDer), paramDer = {paramDer} ; end
    
    net.setParameterServer(parserv, paramDer) ;
  else
    net.setParameterServer(parserv) ;
  end
else
  parserv = [] ;
end

% profile
if params.profile
  if numGpus <= 1
    profile clear ;
    profile on ;
  else
    mpiprofile reset ;
    mpiprofile on ;
  end
end

num = 0 ;
epoch = params.epoch ;
subset = params.(mode) ;
adjustTime = 0 ;

stats.num = 0 ; % return something even if subset = []
stats.time = 0 ;

start = tic ;
for t=1:params.batchSize:numel(subset)
  fprintf('%s: epoch %02d: %3d/%3d:', mode, epoch, ...
          fix((t-1)/params.batchSize)+1, ceil(numel(subset)/params.batchSize)) ;
  batchSize = min(params.batchSize, numel(subset) - t + 1) ;

  for s=1:params.numSubBatches
    % get this image batch and prefetch the next
    batchStart = t + (labindex-1) + (s-1) * numlabs ;
    batchEnd = min(t+params.batchSize-1, numel(subset)) ;
    batch = subset(batchStart : params.numSubBatches * numlabs : batchEnd) ;
    num = num + numel(batch) ;
    if numel(batch) == 0, continue ; end

    inputs = params.getBatch(params.imdb, batch) ;

    if params.prefetch
      if s == params.numSubBatches
        batchStart = t + (labindex-1) + params.batchSize ;
        batchEnd = min(t+2*params.batchSize-1, numel(subset)) ;
      else
        batchStart = batchStart + numlabs ;
      end
      nextBatch = subset(batchStart : params.numSubBatches * numlabs : batchEnd) ;
      params.getBatch(params.imdb, nextBatch) ;
    end

    if isa(net, 'dagnn.DagNN')
      if strcmp(mode, 'train')
        net.mode = 'normal' ;
        net.accumulateParamDers = (s ~= 1) ;
        net.eval(inputs, params.derOutputs, 'holdOn', s < params.numSubBatches) ;
      else
        net.mode = 'test' ;
        net.eval(inputs) ;
      end
    else  % AutoNN
      if strcmp(mode, 'train')
        net.eval(inputs, 'normal', params.derOutputs, s ~= 1) ;
      else
        net.eval(inputs, 'test') ;
      end
    end
  end
  
  % Check if any vars contain NaN or Inf
  if ~isempty(params.onNanInf) && ~isa(net, 'dagnn.DagNN') && ...
    ~all(cellfun(@(x) ~isnumeric(x) || gather(all(isfinite(x(:)))), net.vars))
    switch params.onNanInf
    case 'error'
      error('The network contains NaN or Inf values.') ;
    case 'warning'
      warning('The network contains NaN or Inf values.') ;
    otherwise  % 'debug'/'break'
      net.displayVars() ;
      warning('The network contains NaN or Inf values.') ;
      keyboard ;
    end
  end

  % Accumulate gradient.
  if strcmp(mode, 'train')
    if ~isempty(parserv), parserv.sync() ; end
    if isa(net, 'dagnn.DagNN')
      state = accumulateGradients(net, state, params, batchSize, parserv) ;
    else
      state = accumulateGradientsAutoNN(net, state, params, batchSize, parserv) ;
    end
  end

  % Get statistics.
  time = toc(start) + adjustTime ;
  batchTime = time - stats.time ;
  stats.num = num ;
  stats.time = time ;
  stats = params.extractStatsFn(stats, net, batchSize / max(1, numGpus)) ;
  currentSpeed = batchSize / batchTime ;
  averageSpeed = (t + batchSize - 1) / time ;
  if t == 3*params.batchSize + 1
    % compensate for the first three iterations, which are outliers
    adjustTime = 4*batchTime - time ;
    stats.time = time + adjustTime ;
  end

  fprintf(' %.1f (%.1f) Hz', averageSpeed, currentSpeed) ;
  for f = setdiff(fieldnames(stats)', {'num', 'time'})
    f = char(f) ;
    fprintf(' %s: %.3f', f, stats.(f)) ;
  end
  fprintf('\n') ;

  if params.plotDiagnostics && ~isa(net, 'dagnn.DagNN') && mod(t-1, params.batchSize * 5) == 0
    net.plotDiagnostics(200) ;
  end
end

% Save back to state.
state.stats.(mode) = stats ;
if params.profile
  if numGpus <= 1
    state.prof.(mode) = profile('info') ;
    profile off ;
  else
    state.prof.(mode) = mpiprofile('info');
    mpiprofile off ;
  end
end
if ~params.saveSolverState
  state.solverState = [] ;
else
  for i = 1:numel(state.solverState)
    s = state.solverState{i} ;
    if isnumeric(s)
      state.solverState{i} = gather(s) ;
    elseif isstruct(s)
      state.solverState{i} = structfun(@gather, s, 'UniformOutput', false) ;
    end
  end
end

net.reset() ;
net.move('cpu') ;

% -------------------------------------------------------------------------
function state = accumulateGradients(net, state, params, batchSize, parserv)
% -------------------------------------------------------------------------
numGpus = numel(params.gpus) ;
otherGpus = setdiff(1:numGpus, labindex) ;
if params.normalizedLoss  % assume loss is normalized by batch size
  batchSize = max(1,numGpus) * params.numSubBatches ;
end

for p=1:numel(net.params)

  if ~isempty(parserv)
    parDer = parserv.pullWithIndex(p) ;
  else
    parDer = net.params(p).der ;
  end

  switch net.params(p).trainMethod

    case 'average' % mainly for batch normalization
      thisLR = net.params(p).learningRate ;
      net.params(p).value = vl_taccum(...
          1 - thisLR, net.params(p).value, ...
          (thisLR/batchSize/net.params(p).fanout),  parDer) ;

    case 'gradient'
      thisDecay = params.weightDecay * net.params(p).weightDecay ;
      thisLR = params.learningRate * net.params(p).learningRate ;

      if thisLR>0 || thisDecay>0
        % Normalize gradient and incorporate weight decay.
        parDer = vl_taccum(1/batchSize, parDer, ...
                           thisDecay, net.params(p).value) ;

        if isempty(params.solver)
          % Default solver is the optimised SGD.
          % Update momentum.
          state.solverState{p} = vl_taccum(...
            params.momentum, state.solverState{p}, ...
            -1, parDer) ;

          % Nesterov update (aka one step ahead).
          if params.nesterovUpdate
            delta = params.momentum * state.solverState{p} - parDer ;
          else
            delta = state.solverState{p} ;
          end

          % Update parameters.
          net.params(p).value = vl_taccum(...
            1,  net.params(p).value, thisLR, delta) ;

        else
          % call solver function to update weights
          [net.params(p).value, state.solverState{p}] = ...
            params.solver(net.params(p).value, state.solverState{p}, ...
            parDer, params.solverOpts, thisLR) ;
        end
      end
    otherwise
      error('Unknown training method ''%s'' for parameter ''%s''.', ...
        net.params(p).trainMethod, ...
        net.params(p).name) ;
  end
end

% -------------------------------------------------------------------------
function state = accumulateGradientsAutoNN(net, state, params, batchSize, parserv)
% -------------------------------------------------------------------------

% ensure supported training methods are ordered as expected
assert(isequal(Param.trainMethods, {'gradient', 'average', 'none'})) ;

numGpus = numel(params.gpus) ;
if params.normalizedLoss  % assume loss is normalized by batch size
  batchSize = max(1,numGpus) * params.numSubBatches ;
end

paramVars = [net.params.var] ;
w = net.getValue(paramVars) ;
dw = net.getDer(paramVars) ;
if isscalar(paramVars), w = {w} ; dw = {dw} ; end

if ~params.plotDiagnostics
  % allow memory to be released, for parameters and their derivatives
  net.setValue([paramVars, paramVars + 1], cell(1, 2 * numel(paramVars))) ;
else
  % free only parameter memory, as we still need the gradients for plotting the diagnostics
  net.setValue(paramVars, cell(size(paramVars))) ;
end

for p=1:numel(net.params)
  if ~isempty(parserv)
    parDer = parserv.pullWithIndex(p) ;
  else
    parDer = dw{p} ;
  end

  switch net.params(p).trainMethod
    case 1
      thisDecay = params.weightDecay * net.params(p).weightDecay ;
      thisLR = params.learningRate * net.params(p).learningRate ;

      if thisLR>0 || thisDecay>0
        % Normalize gradient and incorporate weight decay.
        parDer = vl_taccum(1/batchSize, parDer, ...
                           thisDecay, w{p}) ;

        if isempty(params.solver)
          % Default solver is the optimised SGD.
          % Update momentum.
          state.solverState{p} = vl_taccum(...
            params.momentum, state.solverState{p}, ...
            -1, parDer) ;

          % Nesterov update (aka one step ahead).
          if params.nesterovUpdate
            delta = params.momentum * state.solverState{p} - parDer ;
          else
            delta = state.solverState{p} ;
          end

          % Update parameters.
          w{p} = vl_taccum(1, w{p}, thisLR, delta) ;

        else
          % call solver function to update weights
          [w{p}, state.solverState{p}] = ...
            params.solver(w{p}, state.solverState{p}, ...
            parDer, params.solverOpts, thisLR) ;
        end
      end

    case 2 % mainly for batch normalization
      thisLR = net.params(p).learningRate ;
      w{p} = vl_taccum(...
          1 - thisLR, w{p}, ...
          (thisLR/batchSize/net.params(p).fanout),  parDer) ;

    case 3  % none
    otherwise
      error('Unknown training method ''%i'' for parameter ''%s''.', ...
        net.params(p).trainMethod, ...
        net.params(p).name) ;
  end
end

if isscalar(paramVars), w = w{1} ; end
net.setValue(paramVars, w) ;

% -------------------------------------------------------------------------
function stats = accumulateStats(stats_)
% -------------------------------------------------------------------------

for s = {'train', 'val'}
  s = char(s) ;
  total = 0 ;

  % initialize stats stucture with same fields and same order as
  % stats_{1}
  stats__ = stats_{1} ;
  names = fieldnames(stats__.(s))' ;
  values = zeros(1, numel(names)) ;
  fields = cat(1, names, num2cell(values)) ;
  stats.(s) = struct(fields{:}) ;

  for g = 1:numel(stats_)
    stats__ = stats_{g} ;
    num__ = stats__.(s).num ;
    total = total + num__ ;

    for f = setdiff(fieldnames(stats__.(s))', 'num')
      f = char(f) ;
      stats.(s).(f) = stats.(s).(f) + stats__.(s).(f) * num__ ;

      if g == numel(stats_)
        stats.(s).(f) = stats.(s).(f) / total ;
      end
    end
  end
  stats.(s).num = total ;
end

% -------------------------------------------------------------------------
function stats = extractStats(stats, net, sel, ~, ~)
% -------------------------------------------------------------------------
for i = 1:numel(sel)
  stats.(net.layers(sel(i)).outputs{1}) = net.layers(sel(i)).block.average ;
end

% -------------------------------------------------------------------------
function stats = extractStatsAutoNN(stats, net, sel, batchSize, normalizedLoss)
% -------------------------------------------------------------------------
for i = 1:numel(sel)
  name = net.forward(sel(i)).name ;
  if ~isfield(stats, name)
    stats.(name) = 0 ;
  end
  newValue = gather(sum(net.vars{net.forward(sel(i)).outputVar(1)}(:))) ;
  if normalizedLoss  % Undo normalization for following code
    newValue = newValue * batchSize ;
  end
  % Update running average (same work as dagnn.Loss)
  stats.(name) = ((stats.num - batchSize) * stats.(name) + newValue) / stats.num ;
end

% -------------------------------------------------------------------------
function saveState(fileName, net_, state)
% -------------------------------------------------------------------------
net = net_.saveobj() ;
save(fileName, 'net', 'state') ;

% -------------------------------------------------------------------------
function saveStats(fileName, stats)
% -------------------------------------------------------------------------
if exist(fileName)
  save(fileName, 'stats', '-append') ;
else
  save(fileName, 'stats') ;
end

% -------------------------------------------------------------------------
function [net, state, stats] = loadState(fileName)
% -------------------------------------------------------------------------
load(fileName, 'net', 'state', 'stats') ;
if isfield(net, 'layers')
%   net = dagnn.DagNN.loadobj(net) ;
  error('The last checkpoint does not contain a valid AutoNN model.') ;
else
  net = Net(net) ;
end
if isempty(whos('stats'))
  error('Epoch ''%s'' was only partially saved. Delete this file and try again.', ...
        fileName) ;
end

% -------------------------------------------------------------------------
function epoch = findLastCheckpoint(modelDir)
% -------------------------------------------------------------------------
list = dir(fullfile(modelDir, 'net-epoch-*.mat')) ;
tokens = regexp({list.name}, 'net-epoch-([\d]+).mat', 'tokens') ;
epoch = cellfun(@(x) sscanf(x{1}{1}, '%d'), tokens) ;
epoch = max([epoch 0]) ;

% -------------------------------------------------------------------------
function switchFigure(n)
% -------------------------------------------------------------------------
if get(0,'CurrentFigure') ~= n
  try
    set(0,'CurrentFigure',n) ;
  catch
    figure(n) ;
  end
end

% -------------------------------------------------------------------------
function clearMex()
% -------------------------------------------------------------------------
clear vl_tmove vl_imreadjpeg ;

% -------------------------------------------------------------------------
function prepareGPUs(opts, cold)
% -------------------------------------------------------------------------
numGpus = numel(opts.gpus) ;
if numGpus > 1
  % check parallel pool integrity as it could have timed out
  pool = gcp('nocreate') ;
  if ~isempty(pool) && pool.NumWorkers ~= numGpus
    delete(pool) ;
  end
  pool = gcp('nocreate') ;
  if isempty(pool)
    parpool('local', numGpus) ;
    cold = true ;
  end

end
if numGpus >= 1 && cold
  fprintf('%s: resetting GPU\n', mfilename)
  clearMex() ;
  if numGpus == 1
    gpuDevice(opts.gpus)
  else
    spmd
      clearMex() ;
      gpuDevice(opts.gpus(labindex))
    end
  end
end
