function checkLearningParams(mcn_outs, opts)
%CHECKlEARNINGPARAMS compare parameters against caffe.  

  % Algo: we first parse the prototxt and build a set of basic "layer" 
  % objects to store parameters.  These can then be directly compared against
  % their mcn equivalents to reduced the risk of incorrect initialisation.
  caffeLayers = parseCaffeLayers(opts) ;

  % loop over layers and check against network
  for ii = 1:numel(caffeLayers)
    layer = caffeLayers{ii} ;
    msg = 'checking layer settings (%d/%d): %s\n' ;
    fprintf(msg, ii, numel(caffeLayers), layer.name) ;
    ignoreTypes = {'ReLU', 'Scale', 'Silence', 'Eltwise', 'Accuracy', ...
              'BatchNorm', 'ImageData'} ; 
    ignoreNames = {'input-data', 'AnchorTargetLayer', 'rpn-data', ...
                   'roi-data', 'Annotation'} ;
    if ismember(layer.type, ignoreTypes), continue ; end
    if ismember(layer.name, ignoreNames), continue ; end
    mcnLayerName = layer.name ;
    found = false ;
    if contains(layer.name, '-')
      mcnLayerName = strrep(mcnLayerName, '-', '_') ;
      fprintf('renaming search layer %s to %s\n', layer.name, mcnLayerName) ;
    end
    for jj = 1:numel(mcn_outs)
      mcnLayer = mcn_outs{jj}.find(mcnLayerName) ;
      if ~isempty(mcnLayer), mcn = mcnLayer{1} ; found = true ; break ; end
    end
    assert(found, 'matching layer not found') ;
    switch layer.type
      case 'Convolution'
        checkFields = {'stride', 'pad', 'dilate', 'out', 'kernel_size', ...
                       'lr_mult', 'decay_mult'} ;
        hasBias = isfield(layer, 'lr_multx') ; 
        mcnFilters = mcn.inputs{2} ; % assume square filters
        msg = 'code must be modified to handle non-square filter checks' ;
        assert(size(mcnFilters.value,1) == size(mcnFilters.value,2), msg) ;
        filterOpts = {'kernel_size', size(mcnFilters.value, 1), ...
                      'out', size(mcnFilters.value, 4), ...
                      'lr_mult', mcnFilters.learningRate, ...
                      'decay_mult', mcnFilters.weightDecay} ;
        mcnArgs = [ mcn.inputs filterOpts ] ;

        if hasBias
          mcnBias = mcnArgs{3} ; 
          biasOpts = {'lr_multx', mcnBias.learningRate, ...
                      'decay_multx', mcnBias.weightDecay} ;
          mcnArgs = [ mcnArgs biasOpts ] ; %#ok
          checkFields = [checkFields biasOpts([1 3])] ; %#ok
        end
        mcnArgs(strcmp(mcnArgs, 'CuDNN')) = [] ;

        % extract params, fill in defaults and convert to canonical shape
        caffe.stride = fetch(layer, 'stride', [1 2], [1 1]) ;
        caffe.pad = fetch(layer, 'pad', [1 4], [0 0 0 0]) ;
        caffe.out = fetch(layer, 'num_output', 1, 1) ;
        caffe.dilate = fetch(layer, 'dilation', [1 2], [1 1]) ;
        caffe.kernel_size = fetch(layer, 'kernel_size', [1 2], [1 1]) ;
        caffe.decay_mult = fetch(layer, 'decay_mult', 1, 1) ;
        caffe.lr_mult = fetch(layer, 'lr_mult', 1, 1) ;

        if hasBias 
          caffe.lr_multx = fetch(layer, 'lr_multx', 1, 2) ; 
          caffe.decay_multx = fetch(layer, 'decay_multx', 1, 0) ; 
        end
      case 'BatchNorm' 
        checkFields = {'lr_mult', 'lr_multx', 'lr_multxx', ...
                       'decay_mult', 'decay_multx', 'decay_multxx'} ;
        mcnMult = mcn.inputs{2} ; mcnBias = mcn.inputs{3} ; 
        mcnMoments = mcn.inputs{4} ; 
        mcnArgs = {'lr_mult', mcnMult.learningRate, ...
                   'decay_mult', mcnMult.weightDecay, ...
                   'lr_multx', mcnBias.learningRate, ...
                   'decay_multx', mcnBias.weightDecay, ...
                   'lr_multxx', mcnMoments.learningRate, ...
                   'decay_multxx', mcnMoments.weightDecay} ;
        for jj = 1:numel(checkFields)
          caffe.(checkFields{jj}) = str2double(layer.(checkFields{jj})) ;
        end
      case 'Pooling' 
        checkFields = {'stride', 'pad', 'method', 'kernel_size'} ;
        caffe.kernel_size = fetch(layer, 'kernel_size', [1 2], [1 1]) ;
        caffe.stride = fetch(layer, 'stride', [1 2], [1 1]) ;
        caffe.pad = fetch(layer, 'pad', [1 4], [0 0 0 0]) ;
        % different convnetions: mcn `avg` == caffe `ave` (both use 
        % `max` for max pooling
        caffe. method = strrep(lower(layer.pool), 'ave', 'avg') ; 
        poolOpts = mcn.inputs(3:end) ;
        poolOpts(strcmp(poolOpts, 'CuDNN')) = [] ;
        mcnArgs = [poolOpts {'kernel_size', mcn.inputs{2}}] ;
      otherwise, fprintf('checking layer type: %s\n', layer.type) ;
    end
    % run checks
    for jj = 1:numel(checkFields)
      fieldName = checkFields{jj} ;
      mcnPos = find(strcmp(mcnArgs, fieldName)) + 1 ;
      value = mcnArgs{mcnPos} ; cValue = caffe.(fieldName) ;
      if strcmp(fieldName, 'pad')
        % since mcn and caffe handle padding slightly differntly, produce a 
        % warning rather than an error for different padding settings
        if ~all(value == cValue)
          msg = 'WARNING:: pad values do not match for %s: %s vs %s\n' ;
          fprintf(msg, layer.name, mat2str(value), mat2str(cValue)) ;
        end
      else
        msg = sprintf('unmatched parameters for %s', fieldName) ;
        assert(all(value == cValue), msg) ;
      end
    end
  end

% ---------------------------------------------
function x = fetch(layer, name, shape, default)
% ---------------------------------------------
  if isfield(layer, name) 
    x = str2double(layer.(name)) ;
    if numel(x) == 1, x = repmat(x, shape) ; end
  else
    x = default ; 
  end

% --------------------------------------
function layers = parseCaffeLayers(opts)
% --------------------------------------
  % create name map
  nameMap = containers.Map ; 
  nameMap('rpn_conv/3x3') = 'rpn_conv_3x3' ;
  nameMap('psroipooled_loc_rois') = 'psroipooled_bbox_rois' ;
  nameMap('loss') = 'loss_cls' ; % maintain mcn consistency
  proto = fileread(opts.modelOpts.protoPath) ;

  % mini parser
  stack = {} ; tokens = strsplit(proto, '\n') ; 
  known = {'ResNet-50', 'ResNet50_BN_SCALE_Merge', ...
           'VGG_ILSVRC_16_layers', 'SEC'} ;
  msg = 'wrong proto' ; 
  assert(contains(tokens{1}, known), msg) ; tokens(1) = [] ;  
  layers = {} ; layer = struct() ;
  while ~isempty(tokens)
    head = tokens{1} ; tokens(1) = [] ; clean = cleanStr(head) ;
    if isempty(clean) || strcmp(clean(1), '#') 
      % comment or blank proto line (do nothing)
    elseif contains(head, '}') && contains(head, '{') 
      % NOTE: it's not always necessary to flatten out subfields
      pair = strsplit(head, '{') ; key = cleanStr(pair{1}) ; 
      value = strjoin(pair(2:end), '{') ; 
      ptr = numel(value) - strfind(fliplr(value), '}') ; 
      value = value(1:ptr) ;
      ignore = {'reshape_param'} ; % caffe and mcn use different values
      examine = {'param', 'weight_filler', 'bias_filler', 'smooth_l1_loss_param'} ;
      switch key
        case ignore, continue ;
        case examine, pairs = parseInlinePairs(value) ;
        otherwise, error('nested key %s not recognised', key) ;
      end
      for jj = 1:numel(pairs)
        pair = strsplit(pairs{jj}, ':') ; 
        layer = put(layer, cleanStr(pair{1}), cleanStr(pair{2})) ;
      end
    elseif contains(head, '}'), stack(end) = [] ; 
    elseif contains(head, '{'), stack{end+1} = head ; %#ok
    else % handle some messy cases
      tuple = strsplit(head, ':') ; 
      if numel(tuple) > 2
        if strcmp(cleanStr(tuple{1}), 'param_str')
          if numel(tuple) == 3 
            % standard param_str spec form. E.g.
            %   param_str: "'feat_stride': 16"
            tuple(1) = [] ; % pop param_str specifier 
          else, keyboard
          end
        elseif numel(tuple) == 4 
          pairs = parseInlinePairs(head) ;
          for jj = 1:numel(pairs) 
            pair = strsplit(pairs{jj}, ':') ; 
            layer = put(layer, cleanStr(pair{1}), cleanStr(pair{2})) ;
          end
        else, keyboard ; 
        end
      end
      key = cleanStr(tuple{1}) ; value = cleanStr(tuple{2}) ;
      %if contains(head, 'rpn_conv/3x3'), keyboard ; end
      if isKey(nameMap, value), value = nameMap(value) ; end
      layer = put(layer, key, value) ;
    end
    if isempty(stack) && ~isempty(layer)
      layers{end+1} = layer ; layer = {} ; %#ok
    end
  end

% -------------------------------------
function layer = put(layer, key, value)
% -------------------------------------
% store key-value pairs in layer without overwriting previous
% values. Due to MATLAB key naming restrictions, an x-suffix count is used
% for indexing
  while isfield(layer, key), key = sprintf('%sx', key) ; end 
  layer.(key) = value ;

% ------------------------------------
function pairs = parseInlinePairs(str) 
% ------------------------------------
% PARSIiNLINEPAIRS parses prototxt strings in which key-value pairs 
% are supplied in a line, delimited only by whitespace.  For example:
%     kernel_size: 3 pad: 1 stride: 1

  str = strtrim(str) ; % remove leading/trailing whitespace
  dividers = strfind(str, ' ') ; 
  assert(mod(numel(dividers),2) == 1, 'expected odd number of dividers') ;
  starts = [1 dividers(2:2:end)+1] ; 
  ends = [dividers(2:2:end)-1 numel(str)] ;
  pairs = arrayfun(@(s,e) {str(s:e)}, starts, ends) ;

% --------------------------
function str = cleanStr(str)
% --------------------------
% prune unused space and punctuation from prototxt files
  % clean up 
  str = strrep(strrep(strrep(str, '"', ''), ' ', ''), '''', '') ;
  % stop at comments
  comment = strfind(str, '#') ;
  if ~isempty(comment)
    str = str(1:comment(1)-1) ; % stop at first #
  end
