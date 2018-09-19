classdef MNIST < datasets.Dataset
%MNIST MNIST dataset
%   Encapsulates the MNIST dataset for training.
%
%   D = datasets.MNIST('/data/mnist') loads the MNIST dataset from the
%   directory '/data/mnist'. If the data is not found, it is downloaded
%   automatically.
%
%   D.train() returns a cell array with mini-batches, from the shuffled
%   training set. Each mini-batch consists of a set of indexes.
%
%   D.val() returns a cell array with mini-batches, from the validation set
%   (without shuffling). Each mini-batch consists of a set of indexes.
%
%   [IMAGES, LABELS] = D.get(BATCH) returns a tensor of images and the
%   corresponding labels for the given mini-batch BATCH. The images are
%   always mean-centered (by subtracting D.dataMean).
%
%   datasets.MNIST(...,'option', value, ...) sets the following properties:
%
%   `batchSize`:: 128
%     The batch size.
%
%   `partialBatches`:: false
%     Whether partial batches are returned (which can happen for the last
%     batch in a set, if the batch size does not divide the set size).
%
%   See 'autonn/examples/cnn/mnist_example.m' for a full example.

% Copyright (C) 2018 Joao F. Henriques, Andrea Vedaldi.
% All rights reserved.
%
% This file is part of the VLFeat library and is made available under
% the terms of the BSD license (see the COPYING file).

  properties
    images  % images tensor
    dataMean  % image mean
    labels  % labels (1 to 10)
  end
  
  methods
    function dataset = MNIST(dataDir, varargin)
      % parse generic Dataset arguments
      varargin = dataset.parseGenericArgs(varargin) ;
      assert(isempty(varargin), 'Unknown arguments.') ;
      
      % load object from cache if possible
      cache = [dataDir '/mnist.mat'] ;
      if exist(cache, 'file')
        load(cache, 'dataset') ;
      else
        dataset.loadRawData(dataDir) ;
        save(cache, 'dataset') ;
      end
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
      % download and load raw MNIST data
      files = {'train-images-idx3-ubyte', ...
               'train-labels-idx1-ubyte', ...
               't10k-images-idx3-ubyte', ...
               't10k-labels-idx1-ubyte'} ;

      if ~exist(data_dir, 'dir')
        mkdir(data_dir) ;
      end

      for i=1:4
        if ~exist(fullfile(data_dir, files{i}), 'file')
          url = sprintf('http://yann.lecun.com/exdb/mnist/%s.gz',files{i}) ;
          fprintf('downloading %s\n', url) ;
          gunzip(url, data_dir) ;
        end
      end

      f=fopen(fullfile(data_dir, 'train-images-idx3-ubyte'),'r') ;
      x1=fread(f,inf,'uint8') ;
      fclose(f) ;
      x1=permute(reshape(x1(17:end),28,28,60e3),[2 1 3]) ;

      f=fopen(fullfile(data_dir, 't10k-images-idx3-ubyte'),'r') ;
      x2=fread(f,inf,'uint8') ;
      fclose(f) ;
      x2=permute(reshape(x2(17:end),28,28,10e3),[2 1 3]) ;

      f=fopen(fullfile(data_dir, 'train-labels-idx1-ubyte'),'r') ;
      y1=fread(f,inf,'uint8') ;
      fclose(f) ;
      y1=double(y1(9:end)')+1 ;

      f=fopen(fullfile(data_dir, 't10k-labels-idx1-ubyte'),'r') ;
      y2=fread(f,inf,'uint8') ;
      fclose(f) ;
      y2=double(y2(9:end)')+1 ;
      
      o.trainSet = 1 : numel(y1) ;
      o.valSet = numel(y1) + 1 : numel(y1) + numel(y2) ;
      
      im = single(reshape(cat(3, x1, x2),28,28,1,[])) ;
      o.dataMean = mean(im(:,:,:,o.trainSet), 4) ;  % mean-center all images
      o.images = bsxfun(@minus, im, o.dataMean) ;
      o.labels = cat(2, y1, y2) ;
    end
  end
end

