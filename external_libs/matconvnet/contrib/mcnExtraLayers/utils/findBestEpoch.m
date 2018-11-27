function bestEpoch = findBestEpoch(expDir, varargin)
%FINDBESTEPOCH finds the best epoch of training
%   FINDBESTEPOCH(EXPDIR) evaluates the checkpoints
%   (the `net-epoch-%d.mat` files created during
%   training) in EXPDIR
%
%   FINDBESTEPOCH(..., 'option', value, ...) accepts the following
%   options:
%
%   `priorityMetric`:: 'classError'
%    Determines the highest priority metric by which to rank the
%    checkpoints.
%
%   `prune`:: false
%    Removes all saved checkpoints to save space except:
%
%       1. The checkpoint with the lowest validation error metric
%       2. The last checkpoint
%
% Copyright (C) 2017 Samuel Albanie
% Licensed under The MIT License [see LICENSE.md for details]

  opts.prune = false ;
  opts.priorityMetric = 'classError' ;
  opts = vl_argparse(opts, varargin) ;

  lastEpoch = findLastCheckpoint(expDir);
  if ~lastEpoch, return ; end % return if no checkpoints were found

  bestEpoch = findBestValCheckpoint(expDir, opts.priorityMetric);
  if opts.prune
    preciousEpochs = [bestEpoch lastEpoch];
    removeOtherCheckpoints(expDir, preciousEpochs);
    fprintf('----------------------- \n');
    fprintf('directory cleaned: %s\n', expDir);
    fprintf('----------------------- \n');
  end

% -------------------------------------------------------------------------
function removeOtherCheckpoints(expDir, preciousEpochs)
% -------------------------------------------------------------------------
  list = dir(fullfile(expDir, 'net-epoch-*.mat')) ;
  tokens = regexp({list.name}, 'net-epoch-([\d]+).mat', 'tokens') ;
  epochs = cellfun(@(x) sscanf(x{1}{1}, '%d'), tokens) ;
  targets = ~ismember(epochs, preciousEpochs);
  files = cellfun(@(x) fullfile(expDir, sprintf('net-epoch-%d.mat', x)), ...
          num2cell(epochs(targets)), 'UniformOutput', false);
  cellfun(@(x) delete(x), files)

% -------------------------------------------------------------------------
function bestEpoch = findBestValCheckpoint(expDir, priorityMetric)
% -------------------------------------------------------------------------
  lastEpoch = findLastCheckpoint(expDir) ;
  if strcmp(priorityMetric, 'last'), bestEpoch = lastEpoch ; return ; end
  % handle the different storage structures/error metrics
  path = fullfile(expDir, sprintf('net-epoch-%d.mat', lastEpoch)) ;
  try
    data = load(path) ;
  catch
		msg = 'checkopint at %s was malformed, trying agin in 10 secs....\n' ;
		warning(msg, path) ; pause(10) ; data = load(path) ;
  end
  if isfield(data, 'stats')
    valStats = data.stats.val;
  elseif isfield(data, 'info')
    valStats = data.info.val;
  elseif isfield(data, 'state')
    valStats = data.state.stats.val ;
  else
    error('storage structure not recognised');
  end

  ascending = {'mAP', 'accuracy'} ;
  descending = {'top1error', 'error', 'mbox_loss', 'class_loss'} ;

  % find best checkpoint according to the following priority
  metrics = [{priorityMetric} ascending descending] ;

  for i = 1:numel(metrics)
    if isfield(valStats, metrics{i})
      errorMetric = [valStats.(metrics{i})] ;
      selectedMetric = metrics{i} ;
      break ;
    end
  end

  assert(logical(exist('errorMetric', 'var')), 'error metrics not recognized') ;
  if ismember(selectedMetric, ascending)
    pick = @max ;
  else
    pick = @min ;
  end

  [~, bestEpoch] = pick(errorMetric);

% -------------------------------------------------------------------------
function epoch = findLastCheckpoint(expDir)
% -------------------------------------------------------------------------
  list = dir(fullfile(expDir, 'net-epoch-*.mat')) ;
  tokens = regexp({list.name}, 'net-epoch-([\d]+).mat', 'tokens') ;
  epoch = cellfun(@(x) sscanf(x{1}{1}, '%d'), tokens) ;
  epoch = max([epoch 0]) ;
