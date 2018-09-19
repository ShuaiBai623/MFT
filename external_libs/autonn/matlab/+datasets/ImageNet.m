classdef ImageNet < datasets.ImageFolder
%IMAGENET ImageNet dataset
%   Encapsulates the ImageNet dataset for training.
%
%   D = datasets.ImageNet('dataDir', '/data/imagenet') loads the ImageNet
%   dataset from the directory '/data/imagenet'.
%   Detailed instructions on how to set it up are below.
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
%   datasets.ImageNet(...,'option', value, ...) sets several properties,
%   which are all inherited from the datasets.ImageFolder class.
%   See 'help datasets.ImageFolder' for more information.
%
%   See 'autonn/examples/cnn/imagenet_example.m' for a full example.
%
%
%   * ImageNet data setup:
% 
%   The ImageNet/ILSVRC data ships in several TAR archives that can be
%   obtained from the ILSVRC challenge website. You will need:
% 
%   ILSVRC2012_img_train.tar
%   ILSVRC2012_img_val.tar
%   ILSVRC2012_img_test.tar
%   ILSVRC2012_devkit.tar
% 
%   Note that images in the CLS-LOC challenge are the same for the 2012,
%   2013, and 2014 edition of ILSVRC, but that the development kit is
%   different. However, all devkit versions should work.
% 
%   Create the following hierarchy:
% 
%   <dataDir>/images/train/ : content of ILSVRC2012_img_train.tar
%   <dataDir>/images/val/ : content of ILSVRC2012_img_val.tar
%   <dataDir>/images/test/ : content of ILSVRC2012_img_test.tar
%   <dataDir>/ILSVRC2012_devkit : content of ILSVRC2012_devkit.tar
% 
%   In order to speedup training and testing, it is a good idea to
%   preprocess the images to have a fixed size (e.g. 256 pixels high).
%   There is a script in MatConvNet (utils/preprocess-imagenet.sh) that
%   achieves this. It may also be necessary to store the images in RAM
%   disk. Reading images off disk with a sufficient speed is crucial for
%   fast training.

% Copyright (C) 2018 Joao F. Henriques, Andrea Vedaldi.
% All rights reserved.
%
% This file is part of the VLFeat library and is made available under
% the terms of the BSD license (see the COPYING file).

  properties
    labels
    sets
    classes
  end
  
  methods
    function o = ImageNet(varargin)
      % call parent class's constructor, but delay initialization
      o = o@datasets.ImageFolder('skipInitialization', true) ;
      
      % parse generic ImageFolder arguments
      varargin = o.parseGenericArgs(varargin) ;
      assert(isempty(varargin), 'Unknown arguments.') ;
      
      assert(~isempty(o.dataDir), 'Must specify dataset.dataDir.') ;
      
      cache = [o.dataDir '/imagenet.mat'] ;
      if exist(cache, 'file')
        % load from cache file
        s = load(cache) ;
        o.filenames = s.filenames ;
        o.labels = s.labels ;
        o.sets = s.sets ;
        o.classes = s.classes ;
      else
        % use matconvnet's example imagenet imdb as-is (see below)
        imdb = cnn_imagenet_setup_data('dataDir', o.dataDir) ;

        % read info from imdb
        o.filenames = strcat(['images' filesep], imdb.images.name) ;
        o.labels = imdb.images.label ;
        o.sets = imdb.images.set ;
        o.classes = imdb.classes ;
        
        % save to cache file
        [filenames, labels, sets, classes] = deal(o.filenames, o.labels, o.sets, o.classes) ;  %#ok<ASGLU>
        save(cache, 'filenames', 'labels', 'sets', 'classes', '-v6') ;  % no compression, < 2 GB
      end
      
      o.trainSet = find(o.sets == 1) ;
      o.valSet = find(o.sets == 2) ;

      o.initialize() ;
    end
    
    function [images, labels] = get(o, batch)
      % return a single batch of images, with optional prefetching
      [images, idx] = o.get@datasets.ImageFolder(batch) ;
      labels = o.labels(idx) ;
    end
  end
end



function imdb = cnn_imagenet_setup_data(varargin)
% CNN_IMAGENET_SETUP_DATA  Initialize ImageNet ILSVRC CLS-LOC challenge data
%    This function creates an IMDB structure pointing to a local copy
%    of the ILSVRC CLS-LOC data.

opts.dataDir = fullfile('data','imagenet12') ;
opts = vl_argparse(opts, varargin) ;

% -------------------------------------------------------------------------
%                                                  Load categories metadata
% -------------------------------------------------------------------------

d = dir(fullfile(opts.dataDir, '*devkit*')) ;
if numel(d) == 0
  error('Make sure that ILSVRC data is correctly installed in %s', ...
    opts.dataDir) ;
end
devkitPath = fullfile(opts.dataDir, d(1).name) ;

% find metadata
mt = dir(fullfile(devkitPath, 'data', 'meta_clsloc.mat')) ;
if numel(mt) == 0
  mt = dir(fullfile(devkitPath, 'data', 'meta.mat')) ;
end
metaPath = fullfile(devkitPath, 'data', mt(1).name) ;

% find validation images labels
tmp = dir(fullfile(devkitPath, 'data', '*_validation_ground_truth*')) ;
valLabelsPath = fullfile(devkitPath, 'data', tmp(1).name) ;

% find validation images blacklist
tmp = dir(fullfile(devkitPath, 'data', '*_validation_blacklist*')) ;
if numel(tmp) > 0
  valBlacklistPath = fullfile(devkitPath, 'data', tmp(1).name) ;
else
  valBlacklistPath = [] ;
  warning('Could not find validation images blacklist file');
end

fprintf('using devkit %s\n', devkitPath) ;
fprintf('using metadata %s\n', metaPath) ;
fprintf('using validation labels %s\n', valLabelsPath) ;
fprintf('using validation blacklist %s\n', valBlacklistPath) ;

meta = load(metaPath) ;
cats = {meta.synsets(1:1000).WNID} ;
descrs = {meta.synsets(1:1000).words} ;

imdb.classes.name = cats ;
imdb.classes.description = descrs ;
imdb.imageDir = fullfile(opts.dataDir, 'images') ;

% -------------------------------------------------------------------------
%                                                           Training images
% -------------------------------------------------------------------------

fprintf('searching training images ...\n') ;
names = {} ;
labels = {} ;
for d = dir(fullfile(opts.dataDir, 'images', 'train', 'n*'))'
  [~,lab] = ismember(d.name, cats) ;
  ims = dir(fullfile(opts.dataDir, 'images', 'train', d.name, '*.JPEG')) ;
  names{end+1} = strcat([d.name, filesep], {ims.name}) ;
  labels{end+1} = ones(1, numel(ims)) * lab ;
  fprintf('.') ;
  if mod(numel(names), 50) == 0, fprintf('\n') ; end
  %fprintf('found %s with %d images\n', d.name, numel(ims)) ;
end
names = horzcat(names{:}) ;
labels = horzcat(labels{:}) ;

if numel(names) ~= 1281167
  warning('Found %d training images instead of 1,281,167. Dropping training set.', numel(names)) ;
  names = {} ;
  labels =[] ;
end

names = strcat(['train' filesep], names) ;

imdb.images.id = 1:numel(names) ;
imdb.images.name = names ;
imdb.images.set = ones(1, numel(names)) ;
imdb.images.label = labels ;

% -------------------------------------------------------------------------
%                                                         Validation images
% -------------------------------------------------------------------------

ims = dir(fullfile(opts.dataDir, 'images', 'val', '*.JPEG')) ;
names = sort({ims.name}) ;
labels = textread(valLabelsPath, '%d') ;

if numel(ims) ~= 50e3
  warning('Found %d instead of 50,000 validation images. Dropping validation set.', numel(ims))
  names = {} ;
  labels =[] ;
else
  if ~isempty(valBlacklistPath)
    black = textread(valBlacklistPath, '%d') ;
    fprintf('blacklisting %d validation images\n', numel(black)) ;
    keep = setdiff(1:numel(names), black) ;
    names = names(keep) ;
    labels = labels(keep) ;
  end
end

names = strcat(['val' filesep], names) ;

imdb.images.id = horzcat(imdb.images.id, (1:numel(names)) + 1e7 - 1) ;
imdb.images.name = horzcat(imdb.images.name, names) ;
imdb.images.set = horzcat(imdb.images.set, 2*ones(1,numel(names))) ;
imdb.images.label = horzcat(imdb.images.label, labels') ;

% -------------------------------------------------------------------------
%                                                               Test images
% -------------------------------------------------------------------------

ims = dir(fullfile(opts.dataDir, 'images', 'test', '*.JPEG')) ;
names = sort({ims.name}) ;
labels = zeros(1, numel(names)) ;

if numel(labels) ~= 100e3
  warning('Found %d instead of 100,000 test images', numel(labels))
end

names = strcat(['test' filesep], names) ;

imdb.images.id = horzcat(imdb.images.id, (1:numel(names)) + 2e7 - 1) ;
imdb.images.name = horzcat(imdb.images.name, names) ;
imdb.images.set = horzcat(imdb.images.set, 3*ones(1,numel(names))) ;
imdb.images.label = horzcat(imdb.images.label, labels) ;

% -------------------------------------------------------------------------
%                                                            Postprocessing
% -------------------------------------------------------------------------

% sort categories by WNID (to be compatible with other implementations)
[imdb.classes.name,perm] = sort(imdb.classes.name) ;
imdb.classes.description = imdb.classes.description(perm) ;
relabel(perm) = 1:numel(imdb.classes.name) ;
ok = imdb.images.label >  0 ;
imdb.images.label(ok) = relabel(imdb.images.label(ok)) ;

end