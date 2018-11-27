function fix_se_imports(varargin)
%FIX_SE_IMPORTS - clean up imported caffe models
%   FIX_SE_IMPORTS performs some additional clean up work
%   on models imported from caffe to ensure that they are
%   consistent with matconvnet conventions.  It also adds 
%   informaiton about the imagenet dataset used for training to 
%   each model to facilitate easier use in deployment
%
%  TODO: It is much less brittle to try to fix these issues
%  in the caffe import script. The functionality below should be 
%  moved there once the interface is considered stable.
%
% Copyright (C) 2017 Samuel Albanie
% Licensed under The MIT License [see LICENSE.md for details]

  opts.imdbPath = fullfile(vl_rootnn, 'data/imagenet12/imdb.mat') ;
  opts.modelDir = fullfile(vl_rootnn, 'data/models-import') ;
  opts = vl_argparse(opts, varargin) ;

  imdb = load(opts.imdbPath) ;

  % select model
  res = dir(fullfile(opts.modelDir, '*.mat')) ; modelNames = {res.name} ;
  modelNames = modelNames(contains(modelNames, 'SE')) ;

  for mm = 1:numel(modelNames)
    modelPath = fullfile(opts.modelDir, modelNames{mm}) ;
    fprintf('fixing name scheme for %s\n', modelNames{mm}) ;
    net = load(modelPath) ; 

    % fix naming convention
    for ii = 1:numel(net.layers)
      net.layers(ii).name = strrep(net.layers(ii).name, '/', '_') ;
      net.layers(ii).inputs = strrep(net.layers(ii).inputs, '/', '_') ;
      net.layers(ii).outputs = strrep(net.layers(ii).outputs, '/', '_') ;
      net.layers(ii).params = strrep(net.layers(ii).params, '/', '_') ;
    end
    for ii = 1:numel(net.params)
      net.params(ii).name = strrep(net.params(ii).name, '/', '_') ;
    end

    % fix meta 
    fprintf('adding info to %s (%d/%d)\n', modelPath, mm, numel(modelNames)) ;
    net.meta.classes = imdb.classes ;
    net.meta.normalization.imageSize = [224 224 3] ;
    net = dagnn.DagNN.loadobj(net) ; 
    net = net.saveobj() ; save(modelPath, '-struct', 'net') ; %#ok
  end
