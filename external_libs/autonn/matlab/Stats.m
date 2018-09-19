classdef Stats < handle
%STATS Aggregation and plotting of training statistics
%   The Stats class keeps track of values such as objectives and errors
%   during training, and can be used to easily plot them.
%
%   See 'autonn/examples/cnn/mnist_example.m' for a more complete example.
%
%   S = Stats({'var1', 'var2', ...}) creates a Stats object and registers
%   variables with names 'var1', 'var2', etc. These are the variables from
%   a network that need to be plotted, such as objectives and error
%   metrics.
%
%   S.update(NET) fetches the values of all registered variables from
%   object NET (of class Net), and updates the average seen so far. This is
%   normally done for every mini-batch.
%
%   S.update('customstat1', value1, 'customstat2', value2, ...) updates
%   averages of custom statistics, which are not variables in a network.
%   This is useful to create plots of other computations that are done
%   during training, such as custom diagnostics and error metrics.
%
%   S.push('plotname') appends the current averages of all variables to a
%   plot named 'plotname'. The plots can correspond to training phases
%   ('train' or 'val'), but any may be created. Normally this is done at
%   the end of an epoch of training. The averages seen so far are cleared.
%
%   S.print() writes the current statistics to the terminal.
%
%   S.plot() draws the plots in the current figure. See 'help Stats.plot'
%   for more options.
%
%   X = S.values('plotname', 'statname') returns the values of variable
%   'statname' that were stored using S.push('plotname'). This can be used
%   to access the same values that are plotted using S.plot(), for example
%   S.values('train', 'error').

% Copyright (C) 2018 Joao F. Henriques.
% All rights reserved.
%
% This file is part of the VLFeat library and is made available under
% the terms of the BSD license (see the COPYING file).

  properties
    names = {}
    accumulators = []
    counts = []
    lastValues = []
    fromNetwork = []
    
    lookup = struct()
    history = struct()
  end
  
  properties (Transient)
    varIdx = []
    plots = []
  end
  
  methods
    function o = Stats(names)
      % register given stat names
      if nargin > 0
        o.registerVars(names, true) ;
      end
    end
    
    function registerVars(o, names, fromNetwork)
      % register new variables to keep track of (can be given during
      % construction). if fromNetwork is true, they will be fetched
      % automatically from a network when calling update().
      if ischar(names)
        names = {names} ;
      end
      assert(iscellstr(names)) ;
      names = names(:)' ;  % ensure it's a row vector
      zero = zeros(size(names)) ;
      
      offset = numel(o.names) ;
      o.names = [o.names, names] ;
      o.accumulators = [o.accumulators, zero] ;
      o.counts = [o.counts, zero] ;
      o.lastValues = [o.lastValues, zero] ;
      o.varIdx = [o.varIdx, zero] ;
      o.fromNetwork = [o.fromNetwork, zero + fromNetwork] ;
      
      % name-to-index lookup table
      for i = 1:numel(names)
        assert(~isfield(o.lookup, names{i}), 'Variable name already exists.') ;
        o.lookup.(names{i}) = offset + i ;
      end
      
      % insert NaN in existing histories for the new variables
      sets = fieldnames(o.history) ;
      for s = 1:numel(sets)
        o.history.(sets{s})(end + 1 : end + numel(names), :) = NaN ;
      end
    end
    
    function update(o, varargin)
      if isa(varargin{1}, 'Net')
        o.update_net(varargin{1}) ;
        varargin(1) = [] ;
      end
      
      assert(mod(numel(varargin), 2) == 0, 'Expected name-value pairs.') ;
      assert(iscellstr(varargin(1:2:end-1)), 'Expected name-value pairs.') ;
      
      for i = 1:2:numel(varargin)
        o.update_single(varargin{i}, varargin{i + 1}) ;
      end
    end
    
    function v = average(o, name)
      % return current value (average estimate, before push)
      idx = o.lookup.(name) ;
      v = o.accumulators(idx) / o.counts(idx) ;
    end
    
    function v = values(o, set, name)
      % return all values (history created by push) for a set and stat name
      idx = o.lookup.(name) ;
      v = o.history.(set)(idx,:) ;
    end
    
    function reset(o)
      o.accumulators(:) = 0 ;
      o.counts(:) = 0 ;
      o.lastValues(:) = 0 ;
    end
    
    function push(o, set)
      % push current stats into history and reset
      if ~isfield(o.history, set)
        o.history.(set) = [] ;
      end
      o.history.(set)(:,end+1) = o.accumulators ./ o.counts ;
      o.reset() ;
    end
    
    function len = length(o, set)
      % return history length, for a given set
      if isfield(o.history, set)
        len = size(o.history.(set), 2);
      else
        len = 0;
      end
    end
    
    function print(o)
      % print statistics to the terminal
      for i = 1:numel(o.names)
        if o.counts(i) > 0
          name = o.names{i} ;
          v = o.lastValues(i) ;
          avg = o.accumulators(i) / o.counts(i) ;
          if abs(v) < 1e-3 && v ~= 0
            fprintf('%s: %.0e ', name, v) ;
          else
            fprintf('%s: %.3f ', name, v) ;
          end
          if abs(avg) < 1e-3 && avg ~= 0
            fprintf('(avg %.0e) ', avg) ;
          else
            fprintf('(avg %.3f) ', avg) ;
          end
        end
      end
      fprintf('\n') ;
    end
    
    function plot(o, varargin)
%PLOT Plots the statistics on the current figure
%   S.plot('option', value, ...) accepts the following options:
%
%   `figure`:: current
%     Selects a figure number to plot to (e.g. figure 1).
%
%   `names`:: all
%     Specifies a subset of variables to plot (as a cell array of strings).
%
%   `smoothen`:: 0
%     Scale of a Gaussian kernel used to smoothen the plot, 0 to disable it
%     (the default). This can be useful for high-variance/jagged plots.

      opts.figure = [] ;
      opts.names = [] ;
      opts.smoothen = 0 ;
      opts = vl_argparse(opts, varargin) ;
      
      if isempty(opts.names)
        valid_stats = 1:numel(o.names) ;
      else
        valid_stats = find(ismember(o.names, opts.names)) ;
      end
      
      sets = fieldnames(o.history) ;
      
      % create plot objects if it's the first datapoint to be plotted, or
      % the graphics objects have been deleted
      if size(o.history.(sets{1}), 2) == 1 || isempty(o.plots) || ~all(ishandle(o.plots(:)))
        if ~isempty(opts.figure)
          figure(opts.figure) ;
        end
        clf ;
        ax = gobjects(numel(valid_stats), 1) ;
        o.plots = gobjects(numel(valid_stats), numel(sets)) ;
        
        for i = 1:numel(valid_stats)
          ax(i) = axes() ;
          o.plots(i,:) = plot(NaN(2, numel(sets)), '.-', 'Parent', ax(i)) ;
          
          name = o.names{valid_stats(i)} ;
          xlabel('Epoch') ;
          title(name) ;
          legend(sets{:}) ;
          grid on ;
        end
        
        dynamic_subplot(ax) ;
      end
      
      % smoothing kernel
      if opts.smoothen > 0
        window = fspecial('gaussian', [1, ceil(2 * opts.smoothen + 1)], opts.smoothen) ;
        n = size(o.history.(sets{1}), 2);
        normalization = conv(ones(1, n), window, 'same');
      end
      
      % update plots
      for i = 1:numel(valid_stats)
        for s = 1:numel(sets)
          v = o.history.(sets{s})(valid_stats(i),:) ;
          if opts.smoothen > 0
            v = conv(v, window, 'same') ./ normalization;
          end
          set(o.plots(i,s), 'XData', (1:numel(v))', 'YData', v');
        end
      end
      
      drawnow ;
    end
  end
  
  methods (Access = protected)
    function update_net(o, net)
      if isempty(o.varIdx)
        o.varIdx = zeros(size(o.names)) ;
      end
      
      for i = find(o.fromNetwork)
        name = o.names{i} ;
        if o.varIdx(i) == 0  % cache variable index
          o.varIdx(i) = net.getVarIndex(name) ;
        end
        
        % get average of tensor values (possibly from GPU)
        v = net.getValue(o.varIdx(i)) ;
        v = gather(sum(v(:))) / numel(v) ;
        o.lastValues(i) = v ;
        
        % accumulate
        o.accumulators(i) = o.accumulators(i) + v ;
        o.counts(i) = o.counts(i) + 1 ;
      end
    end
    
    function update_single(o, name, v)
      if ~isfield(o.lookup, name)
        % add new variable
        o.registerVars({name}, false) ;
      end
      
      idx = o.lookup.(name) ;
      v = gather(sum(v(:))) / numel(v) ;
      o.lastValues(idx) = v ;
      o.accumulators(idx) = o.accumulators(idx) + v ;
      o.counts(idx) = o.counts(idx) + 1 ;
    end
  end
  
end

