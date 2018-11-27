function [net, info] = cnn_imagenet_se_mcn(varargin)
% CNN_IMAGENET_SE_MCN Evaluate imported PyTorch models on ImageNet val set
% (closely based on the mcn cnn_imagenet.m example)
%
% Copyright (C) 2017 Samuel Albanie
% Licensed under The MIT License [see LICENSE.md for details]

  opts.gpus = 3 ;
  opts.continue = 1 ;
  opts.batchSize = 256 ;
  opts.model = 'SE-ResNet-50-mcn' ;
  opts.modelDir = fullfile(vl_rootnn, 'data/models-import') ;
  opts.dataDir = fullfile(vl_rootnn, 'data/datasets/ILSVRC2012') ;
  opts.labelMap = fullfile(vl_rootnn, 'contrib/mcnSENets/misc/label_map.txt') ;
  [opts, varargin] = vl_argparse(opts, varargin) ;

  opts.expDir = fullfile(vl_rootnn, 'data', ['imagenet12-' opts.model]) ;
  [opts, varargin] = vl_argparse(opts, varargin) ;

  opts.lite = false ;
  opts.numFetchThreads = 12 ;
  opts.imdbPath = fullfile(vl_rootnn, 'data', 'imagenet12', 'imdb.mat');
  opts = vl_argparse(opts, varargin) ;

  opts.train.gpus = opts.gpus ;

% -------------------------------------------------------------------------
%                                                              Prepare data
% -------------------------------------------------------------------------

  if exist(opts.imdbPath, 'file')
    imdb = load(opts.imdbPath) ;
    imdb.imageDir = fullfile(opts.dataDir, 'images');
  else
    imdb = cnn_imagenet_setup_data('dataDir', opts.dataDir, 'lite', opts.lite) ;
    mkdir(opts.expDir) ;
    save(opts.imdbPath, '-struct', 'imdb') ;
  end

  % remap labels to match the order used in training
  imdb = updateLabelMap(imdb, opts) ;

% -------------------------------------------------------------------------
%                                                             Prepare model
% -------------------------------------------------------------------------
  net = load_model(opts.modelDir, opts.model) ;

  % modify the imdb to skip training images
  imdb.images.set(imdb.images.set == 1) = 4 ;
  [net, info] = cnn_train_dag(net, imdb, getBatchFn(opts, net.meta), ...
                              'expDir', opts.expDir, 'gpus', opts.train.gpus, ...
                              'numEpochs', 1, 'continue', opts.continue, ...
                              'batchSize', opts.batchSize) ;

  % hack the checkpoint file to save an empty dag
  modelPath = fullfile(opts.expDir, 'net-epoch-1.mat') ; tmp = load(modelPath) ;
  net_ = net ; net = dagnn.DagNN().saveobj() ; stats = tmp.stats ; state = {} ; %#ok
  save(modelPath, 'net', 'stats', 'state') ; net = net_ ;

% -------------------------------------------------------------------------
function dag = load_model(modelDir, name)
% -------------------------------------------------------------------------
  modelPath = fullfile(modelDir, sprintf('%s.mat', name)) ;
  if ~exist(modelDir, 'dir') , mkdir(modelDir) ; end

  if ~exist(modelPath, 'file')
    fprintf('Downloading the %s model ... this may take a while\n', name) ;
    base = 'http://www.robots.ox.ac.uk/~albanie' ;
    url = sprintf('%s/models/se-nets/%s.mat', base, name) ;
    urlwrite(url, modelPath) ;
  end

  dag = dagnn.DagNN.loadobj(load(modelPath)) ;
  dag.addLayer('softmax', dagnn.SoftMax(), dag.layers(end).outputs, 'prediction', {}) ;
  dag.addLayer('top1err', dagnn.Loss('loss', 'classerror'), ...
               {'prediction','label'}, 'top1err') ;
  dag.addLayer('top5err', dagnn.Loss('loss', 'topkerror', 'opts', {'topK',5}), ...
               {'prediction','label'}, 'top5err') ;

% -------------------------------------------------------------------------
function fn = getBatchFn(opts, meta)
% -------------------------------------------------------------------------
  bopts = struct('useGpu', numel(opts.train.gpus) > 0, ...
                 'imageSize', meta.normalization.imageSize(1:2)) ;
  fn = @(x,y) eval_get_batch(bopts,x,y) ;

% -------------------------------------------------------------------------
function varargout = eval_get_batch(opts, imdb, batch)
% -------------------------------------------------------------------------
  images = strcat([imdb.imageDir filesep], imdb.images.name(batch)) ;
  data = getImageBatch(images, opts, 'prefetch', nargout == 0) ;
  if nargout > 0
    labels = imdb.images.label(batch) ;
    varargout{1} = {'data', data, 'label', labels} ;
  end

% ----------------------------------------
function imdb = updateLabelMap(imdb, opts) 
% ----------------------------------------
  labelMap = importdata(opts.labelMap) ; 
  keep = imdb.images.label ~= 0 ;
  newLabels = labelMap(imdb.images.label(keep)) ;
  imdb.images.label(keep) = newLabels ;

% -------------------------------------------------
function data = getImageBatch(imagePaths, varargin)
% -------------------------------------------------
% GETIMAGEBATCH  Load and jitter a batch of images
opts.useGpu = false ;
opts.prefetch = false ;
opts.numThreads = 10 ;

% Options that were used during PyTorch training
% Note: that normalisation must occur after the pixel
% values have been rescaled to [0,1]
opts.cropSize = 224 / 256 ;
opts.imageSize = [224, 224] ;
opts.meanImg = [123, 117, 104] ;
%opts.std = [0.229, 0.224, 0.225] ;
opts = vl_argparse(opts, varargin);

args{1} = {imagePaths, ...
           'NumThreads', opts.numThreads, ...
           'Pack', ...
           'Interpolation', 'bilinear', ... % use bilinear to reproduce trainig resize
           'Resize', opts.imageSize(1:2), ...
           'CropSize', opts.cropSize, ...
           'CropAnisotropy', [1 1], ... % preserve aspect ratio
           'CropLocation', 'center'} ; % centre crop for testing

if opts.useGpu, args{end+1} = {'Gpu'} ; end
args = horzcat(args{:}) ;

if opts.prefetch
  vl_imreadjpeg(args{:}, 'prefetch') ;
  data = [] ;
else
  data = vl_imreadjpeg(args{:}) ;
  data = bsxfun(@minus, data{1}, permute(opts.meanImg, [1 3 2])) ;
  %data = data{1} / 255 ; % scale to (almost) [0,1]
  %data = bsxfun(@rdivide, data, permute(opts.std, [1 3 2])) ;
end
