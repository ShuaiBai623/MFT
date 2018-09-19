function outputs = pretrained(modelName, varargin)
%PRETRAINED Loads a pre-trained model, possibly downloading it
%   M = models.pretrained(NAME) returns a pre-trained deep network from a
%   local models directory. If the model is not found, it attempts to
%   download it from an online repository.
%
%   The model is returned as a cell array of Layer objects, each one
%   corresponding to one of the network's outputs.
%
%   The MatConvNet repository's list of models is available at:
%   http://www.vlfeat.org/matconvnet/pretrained/
%
%   A more complete repository of models is maintained by Sam Albanie:
%   http://www.robots.ox.ac.uk/~albanie/mcn-models.html
%
%   Notes:
%   - Some less standard models require custom layers, which have to be
%   installed separately (the above websites have further instructions).
%   - If the model is stored in a subdirectory of the repository, then it
%   must be specified (e.g. models.pretrained('multipose/multipose-coco')).
%   You can check if this is the case in the website's download URL.
%
%   M = models.pretrained(NAME, 'modelsDir', DIR) changes the local models
%   directory. The default is '<matconvnet>/data/models'.
%
%   M = models.pretrained(NAME, 'modelsUrls', LIST) changes the remote
%   models repositories, given as a cell array of strings. By default,
%   MatConvNet's and Sam Albanie's repositories are used.

% Copyright (C) 2018 Samuel Albanie, Joao F. Henriques.
% All rights reserved.
%
% This file is part of the VLFeat library and is made available under
% the terms of the BSD license (see the COPYING file).

  opts.modelsDir = [vl_rootnn() '/data/models'] ;
  opts.modelsUrls = {'http://www.vlfeat.org/matconvnet/models', ...
                    'http://www.robots.ox.ac.uk/~albanie/models'} ;
  opts.customLayerFn = [] ;
  opts = vl_argparse(opts, varargin, 'nonrecursive') ;
  
  assert(exist(opts.modelsDir, 'dir') ~= 0, 'Models directory does not exist.') ;
  
  modelPath = [opts.modelsDir '/' modelName '.mat'] ;
  
  % if modelName is in a subdirectory, create it
  if any(modelName == '/')
    mkdir(fileparts(modelPath)) ;
  end
  
  % download if it doesn't exist
  if ~exist(modelPath, 'file')
    for i = 1:numel(opts.modelsUrls)
      url = [opts.modelsUrls{i} '/' modelName '.mat'] ;
      disp(['Model not found; attempting to download from: ' url]) ;
      try
        websave(modelPath, url) ;
      catch
        continue  % try next repository
      end
      break  % no error, we're done
    end
  end
  
  assert(exist(modelPath, 'file') ~= 0, ...
    'Could not download model from known repositories.') ;
  
  % load and convert to AutoNN layer
  net = load(modelPath) ;
  outputs = Layer.fromDagNN(net, opts.customLayerFn) ;

end
