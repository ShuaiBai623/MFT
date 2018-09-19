classdef Dataset < handle
%DATASET Parent class for datasets (e.g. datasets.CIFAR10)
%   The datasets.Dataset class makes it easier to partition a dataset into
%   training/validation sets, as well as in mini-batches.
%
%   It can be used either as a base class (e.g. datasets.CIFAR10), or on
%   its own for custom datasets, without creating a subclass. See the
%   'autonn/examples/cnn/' directory for different examples.
%
%   D = datasets.Dataset() creates an empty, default dataset.
%
%   D.train() returns a cell array with mini-batches, from the shuffled
%   training set. Each mini-batch consists of a set of indexes drawn from
%   the vector D.trainSet.
%
%   D.val() returns a cell array with mini-batches, from the validation set
%   (without shuffling). Each mini-batch consists of a set of indexes drawn
%   from the vector D.valSet.
%
%   D.partition(IDX) partitions a given set (vector) IDX into mini-batches.
%
%   datasets.Dataset('option', value, ...) sets the following properties:
%
%   `batchSize`:: 128
%     The batch size.
%
%   `partialBatches`:: false
%     Whether partial batches are returned (which can happen for the last
%     batch in a set, if the batch size does not divide the set size).
%
%   `trainSet`:: []
%     The training set. Note that subclasses (e.g. datasets.CIFAR10)
%     usually override this property with the default training set.
%
%   `valSet`:: []
%     The validation set. Note that subclasses (e.g. datasets.CIFAR10)
%     usually override this property with the default validation set.

% Copyright (C) 2018 Joao F. Henriques.
% All rights reserved.
%
% This file is part of the VLFeat library and is made available under
% the terms of the BSD license (see the COPYING file).

  properties
    batchSize = 128  % batch size
    trainSet = []  % indexes of training samples
    valSet = []  % indexes of validation samples
    partialBatches = false  % whether to return a partial batch (if the batch size does not divide the dataset size)
  end
  
  methods
    function dataset = Dataset(varargin)
      % parse generic Dataset arguments on non-subclassed construction.
      % allows Dataset to be used as a stand-alone class.
      varargin = dataset.parseGenericArgs(varargin) ;
      assert(isempty(varargin), 'Unknown arguments.') ;
    end
    
    function args = parseGenericArgs(o, args)
      % called by subclasses to parse generic Dataset arguments
      args = vl_parseprop(o, args, {'batchSize', 'trainSet', 'valSet', 'partialBatches'}) ;
    end
    
    function batches = train(o)
      % shuffle training set, and partition it into batches
      assert(~isempty(o.trainSet), ['To use the default dataset.train() ' ...
        'method, the training set must be specified (dataset.trainSet).']) ;
      
      batches = o.partition(o.trainSet(randperm(end))) ;
    end
    
    function batches = val(o)
      % partition validation set into batches
      assert(~isempty(o.valSet), ['To use the default dataset.val() ' ...
        'method, the validation set must be specified (dataset.valSet).']) ;
      
      batches = o.partition(o.valSet) ;
    end
    
    function batches = partition(o, idx, batchSz)
      % partition indexes into batches (stored in a cell array).
      % if IDX is a matrix, each column is a distinct sample.
      if nargin < 3  % allow overriding batch size
        batchSz = o.batchSize ;
      end
      if isvector(idx)
        idx = idx(:)' ;  % ensure row-vector
      end
      batchSz = min(batchSz, size(idx,2));  % guard against big batch size
      batches = cell(1, ceil(size(idx,2) / batchSz)) ;
      b = 1 ;
      for start = 1 : batchSz : size(idx,2)
        batches{b} = idx(:, start : min(start + batchSz - 1, end)) ;
        b = b + 1 ;
      end
      
      % delete last partial batch if needed
      if ~o.partialBatches && size(batches{end}, 2) < batchSz
        batches(end) = [] ;
      end
    end
  end
  
end

