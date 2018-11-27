function compute_label_map(varargin)
% COMPUTE_LABEL_MAP Determines the ILVSRC mapping used in SENets
%   COMPUTE_LABEL_MAP aims to invert the ILSVRC mapping from the 
%   order used to train the public SENEt models to the standard
%   ASCII-ordering convention
%
% Copyright (C) 2017 Samuel Albanie
% Licensed under The MIT License [see LICENSE.md for details]

  opts.numClasses = 1000 ;
  opts.dataDir = fullfile(vl_rootnn, 'data/datasets/ILSVRC2012') ;
  opts.miscDir = fullfile(vl_rootnn, 'contrib/mcnSENets/misc') ;
  opts = vl_argparse(opts, varargin) ;

  imdbPath = fullfile(vl_rootnn, 'data', 'imagenet12', 'imdb.mat');

  % set paths to source and storage destination for label map
  valFile = fullfile(opts.miscDir, 'ILSVRC2017_val.txt') ;
  mapFile = fullfile(opts.miscDir, 'label_map.txt') ;

  if exist(imdbPath, 'file')
    imdb = load(imdbPath) ;
    imdb.imageDir = fullfile(opts.dataDir, 'images');
  else
    imdb = cnn_imagenet_setup_data('dataDir', opts.dataDir, 'lite', opts.lite) ;
    mkdir(opts.expDir) ;
    save(imdbPath, '-struct', 'imdb') ;
  end

  % read examples
  fid = fopen(valFile, 'r') ; data = textscan(fid, '%s %d') ; fclose(fid) ;
  gtImNames = data{1} ; gtLabels = data{2} + 1 ; % fix offset

  % trim stems for comparison
  valIdx = imdb.images.set == 2 ; valNames = imdb.images.name(valIdx) ; 
  imNames = cellfun(@getImName, valNames, 'Uni', 0) ; 
  valLabels = imdb.images.label(valIdx) ; 

  % create map - favour safety over speed here
  labelMap = zeros(1, opts.numClasses) ;
  for ii = 1:numel(imNames)
    imName = imNames{ii} ;
    [member, idx] = ismember(imName, gtImNames) ;
    assert(member, 'unrecognised val image') ;
    srcLabel = valLabels(ii) ; destLabel = gtLabels(idx) ;
    if labelMap(srcLabel) ~= 0 
      assert(labelMap(srcLabel) == destLabel, 'inconsistent label map') ;
    else
      labelMap(srcLabel) = destLabel ;
      fprintf('(%d/%d) remapping %d\n', ii, numel(imNames), srcLabel) ;
    end
  end

  % check for completeness
  assert(numel(unique(labelMap)) == numel(labelMap), 'incomplete mapping') ;
 
  % store label map to disk
  fid = fopen(mapFile, 'w') ; fprintf(fid, '%d\n', labelMap) ; fclose(fid) ;

% -------------------------------
function imName = getImName(path)
% -------------------------------
  [~,stem] = fileparts(path) ;
  imName = strcat(stem, '.JPEG') ;
