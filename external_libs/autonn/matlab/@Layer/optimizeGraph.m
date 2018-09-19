function rootLayer = optimizeGraph(rootLayer)
%OPTIMIZEGRAPH Do graph optimizations, such as merging redundant vl_nnwsum

  % find all weighted sum layers (vl_nnwsum)
  wsums = rootLayer.find(@vl_nnwsum) ;
  
  % decide whether each wsum can be merged or not (unless explicitly set by
  % user, with layer.optimize = true/false). if a wsum is used as input to
  % a *single* other wsum, then it can be merged; if it is used as input to
  % any other layer, or more than 1 wsum, then it cannot be merged.
  allLayers = rootLayer.find() ;
  
  for i = 1:numel(allLayers)
    inputs = allLayers{i}.inputs ;
    
    if isequal(allLayers{i}.func, @vl_nnwsum)
      % a wsum, flag its input wsums; if it was flagged before, then it
      % cannot be merged (because it's used as input to 2 or more wsums).
      for j = 1:numel(inputs)
        in = inputs{j} ;
        if isa(in, 'Layer') && isequal(in.func, @vl_nnwsum) && ~islogical(in.optimize)
          if isempty(in.optimize)
            in.optimize = 'flag' ;
          elseif isequal(in.optimize, 'flag')
            in.optimize = false ;
          end
        end
      end
    else
      % not a wsum, its input wsums cannot be merged unless set explicitly
      for j = 1:numel(inputs)
        in = inputs{j} ;
        if isa(in, 'Layer') && isequal(in.func, @vl_nnwsum) && ~islogical(in.optimize)
          in.optimize = false ;
        end
      end
    end
  end
  
  % remove flags
  for i = 1:numel(wsums)
    if isequal(wsums{i}.optimize, 'flag')
      wsums{i}.optimize = true ;
    end
  end
  
  
  % now optimize each wsum, by merging any of its valid wsum inputs
  for i = 1:numel(wsums)
    inputs = wsums{i}.inputs ;
    
    % make sure there's a 'weights' property at the end, with correct size
    assert(strcmp(inputs{end-1}, 'weights') && ...
      numel(inputs{end}) == numel(inputs) - 2) ;

    % separate inputs to the sum, and weights
    origWeights = inputs{end} ;
    inputs = inputs(1:end-2) ;
    weights = cell(size(inputs)) ;

    for k = 1:numel(inputs)
      in = inputs{k} ;
      if isa(in, 'Layer') && isequal(in.func, @vl_nnwsum) && in.optimize
        % merge weights and store results
        inputs{k} = in.inputs(1:end-2) ;
        weights{k} = origWeights(k) * in.inputs{end} ;
      else
        % any other input (Layer or constant), wrap it in a single cell
        inputs{k} = {in} ;
        weights{k} = origWeights(k) ;
      end
    end

    % merge the results in order
    inputs = [inputs{:}] ;
    weights = [weights{:}] ;

    % store the merged inputs list, and the weights as a name-value pair
    wsums{i}.enableCycleChecks = false ;  % faster
    wsums{i}.inputs = [inputs, {'weights', weights}] ;
    wsums{i}.enableCycleChecks = true ;
  end

end

