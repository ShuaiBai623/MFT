classdef CIFAR10 < datasets.Dataset
%CIFAR10 CIFAR-10 dataset
%   Encapsulates the CIFAR-10 dataset for training.
%
%   D = datasets.CIFAR10('/data/cifar') loads the CIFAR-10 dataset from the
%   directory '/data/cifar'. If the data is not found, it is downloaded
%   automatically.
%
%   D.train() returns a cell array with mini-batches, from the shuffled
%   training set. Each mini-batch consists of a set of indexes.
%
%   D.val() returns a cell array with mini-batches, from the validation set
%   (without shuffling). Each mini-batch consists of a set of indexes.
%
%   [IMAGES, LABELS] = D.get(BATCH) returns a tensor of images and the
%   corresponding labels for the given mini-batch BATCH.
%
%   datasets.CIFAR10(..., 'option', value, ...) sets the following
%   properties:
%
%   `batchSize`:: 128
%     The batch size.
%
%   `contrastNormalization`:: true
%     Whether to apply contrast normalization, as suggested by:
%     Coates et al., "An Analysis of Single-Layer Networks in Unsupervised
%     Feature Learning", PMLR 2011.
%
%   `whitenData`:: true
%     Whether to whiten the data (not including mean-centering, which is
%     only performed if contrastNormalization = true).
%
%   `partialBatches`:: false
%     Whether partial batches are returned (which can happen for the last
%     batch in a set, if the batch size does not divide the set size).
%
%   See 'autonn/examples/cnn/cifar_example.m' for a full example.

% Copyright (C) 2018 Joao F. Henriques, Andrea Vedaldi.
% All rights reserved.
%
% This file is part of the VLFeat library and is made available under
% the terms of the BSD license (see the COPYING file).

  properties
    images  % images tensor
    dataMean  % image mean
    labels  % labels (1 to 10)
    labelNames  % string name associated with each numeric label
    
    contrastNormalization = true
    whitenData = true
  end
  
  methods
    function o = CIFAR10(dataDir, varargin)
      % parse generic Dataset arguments
      varargin = o.parseGenericArgs(varargin) ;
      
      % parse arguments specific to this dataset
      vl_parseprop(o, varargin, {'contrastNormalization', 'whitenData'}) ;
      
      % load object from cache if it exists, and uses the same arguments
      cache = [dataDir '/cifar10.mat'] ;
      if exist(cache, 'file')
        s = load(cache, 'dataset') ;
        if s.dataset.whitenData == o.whitenData && ...
         s.dataset.contrastNormalization == o.contrastNormalization
          
          o = s.dataset ;
          return
        end
      end
      
      % otherwise, create from scratch
      o.loadRawData(dataDir) ;
      dataset = o ;  %#ok<NASGU>
      save(cache, 'dataset') ;
    end
    
    function [images, labels] = get(o, idx)
      % return a single batch (may be wrapped in a cell)
      if iscell(idx) && isscalar(idx)
        idx = idx{1} ;
      end
      images = o.images(:,:,:,idx) ;
      labels = o.labels(idx) ;
    end
  end
  
  methods (Access = protected)
    function loadRawData(o, data_dir)
      % download and load raw CIFAR10 data
      unpackPath = fullfile(data_dir, 'cifar-10-batches-mat');
      files = [arrayfun(@(n) sprintf('data_batch_%d.mat', n), 1:5, 'UniformOutput', false) ...
        {'test_batch.mat'}];
      files = cellfun(@(fn) fullfile(unpackPath, fn), files, 'UniformOutput', false);
      file_set = uint8([ones(1, 5), 3]);

      if ~exist(data_dir, 'dir')
        mkdir(data_dir) ;
      end
      
      if any(cellfun(@(fn) ~exist(fn, 'file'), files))
        url = 'http://www.cs.toronto.edu/~kriz/cifar-10-matlab.tar.gz' ;
        fprintf('downloading %s\n', url) ;
        untar(url, data_dir) ;
      end

      data = cell(1, numel(files));
      labels = cell(1, numel(files));  %#ok<*PROPLC>
      sets = cell(1, numel(files));
      for fi = 1:numel(files)
        fd = load(files{fi}) ;
        data{fi} = permute(reshape(fd.data',32,32,3,[]),[2 1 3 4]) ;
        labels{fi} = fd.labels' + 1; % Index from 1
        sets{fi} = repmat(file_set(fi), size(labels{fi}));
      end

      set = cat(2, sets{:});
      data = single(cat(4, data{:}));
      
      % train and validation sample indexes
      o.trainSet = find(set == 1) ;
      o.valSet = find(set == 3) ;

      % remove mean in any case
      o.dataMean = mean(data(:,:,:,o.trainSet), 4);
      data = bsxfun(@minus, data, o.dataMean);

      % normalize by image mean and std as suggested in `An Analysis of
      % Single-Layer Networks in Unsupervised Feature Learning` Adam
      % Coates, Honglak Lee, Andrew Y. Ng

      if o.contrastNormalization
        z = reshape(data,[],60000) ;
        z = bsxfun(@minus, z, mean(z,1)) ;
        n = std(z,0,1) ;
        z = bsxfun(@times, z, mean(n) ./ max(n, 40)) ;
        data = reshape(z, 32, 32, 3, []) ;
      end

      if o.whitenData
        z = reshape(data,[],60000) ;
        W = z(:,o.trainSet)*z(:,o.trainSet)'/60000 ;
        [V,D] = eig(W) ;
        % the scale is selected to approximately preserve the norm of W
        d2 = diag(D) ;
        en = sqrt(mean(d2)) ;
        z = V*diag(en./max(sqrt(d2), 10))*V'*z ;
        data = reshape(z, 32, 32, 3, []) ;
      end
      
      names = load(fullfile(unpackPath, 'batches.meta.mat'));

      o.images = data ;
      o.labels = single(cat(2, labels{:})) ;
      o.labelNames = names.label_names;
    end
  end
end

