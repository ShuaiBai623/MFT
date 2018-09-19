function varargout = While(iteration, varargin)
%WHILE Differentiable While-loop or recursion, with dynamic stop condition

% Copyright (C) 2017 Joao F. Henriques.
% All rights reserved.
%
% This file is part of the VLFeat library and is made available under
% the terms of the BSD license (see the COPYING file).


  opts.concatenate = [] ;  % if non-empty, outputs are concatenated along specified dimensions
  opts.stopCondition = true ;  % whether a stop condition is output by the iteration function
  opts.count = inf ;  % maximum number of iterations, even if the stop condition wasn't achieved
  [opts, initial_values] = vl_argparsepos(opts, varargin, 'nonrecursive') ;
  
  assert(opts.stopCondition || ~isequal(opts.count, inf), ...
    'Must specify either a stop condition or count, to prevent infinite loops.') ;
  
  assert(nargin(iteration) > 1 || nargin(iteration) < 0, ...
    'The iteration function must accept at least 2 arguments (1 recursion variable and the counter).') ;
  
  assert(nargin(iteration) - 1 == numel(initial_values), ...
    'Must specify an initial value for each recursion variable (input/output of iteration function).') ;


  % the number of recursion variables (i.e., inputs/outputs of recursion
  % function)
  num_rec_vars = numel(initial_values) ;
  
  % execute function on new Inputs (for recursive variables, plus counter)
  inputs = cell(1, num_rec_vars + 1) ;
  for i = 1:num_rec_vars
    inputs{i} = Input(sprintf('while_input_r%i', i)) ;
  end
  inputs{num_rec_vars + 1} = Input('while_input_counter') ;
  
  outputs = cell(1, num_rec_vars) ;
  [outputs{:}] = iteration(inputs{:}) ;

  % gather all layers
  layers = find(cat(1, outputs{:})) ;

  % get anonymous function's static workspace (non-recursive variables)
  info = functions(iteration) ;
  assert(strcmp(info.type, 'anonymous'), 'The first argument must be an anonymous function.') ;
  workspace = info.workspace{1} ;
  names = fieldnames(workspace) ;
  
  non_rec_vars = cell(1, numel(names)) ;
  num_non_rec_vars = 0 ;
  
  for i = 1:numel(names)
    % replace corresponding layers with simple Inputs
    old_layer = workspace.(names{i}) ;
    if isa(old_layer, 'Layer')
      num_non_rec_vars = num_non_rec_vars + 1 ;
      non_rec_vars{num_non_rec_vars} = old_layer ;  % remember old layer
      
      new_layer = Input(sprintf('while_input_n%i', num_non_rec_vars)) ;
      
      % do the replacement in the inputs lists of all layers
      for j = 1:numel(layers)
        for k = 1:numel(layers{j}.inputs)
          if eq(old_layer, layers{j}.inputs{k}, 'sameInstance')
            layers{j}.inputs{k} = new_layer ;
          end
        end
      end
    end
  end

  % compile iteration network
  net = Net(outputs{:}) ;
  
  
  % cache indexes of variables in iteration network's meta information
  net.meta.rec_var = zeros(1, num_rec_vars) ;  % recursive vars
  for i = 1:num_rec_vars
    net.meta.rec_var(i) = net.getVarIndex(sprintf('while_input_r%i', i)) ;
  end
  net.meta.non_rec_var = zeros(1, num_non_rec_vars) ;  % non-recursive vars
  for i = 1:num_non_rec_vars
    net.meta.non_rec_var(i) = net.getVarIndex(sprintf('while_input_n%i', i)) ;
  end
  net.meta.counter_var = net.getVarIndex('while_input_counter', false) ;  % 0 if no counter
  net.meta.output_var = net.getVarIndex(outputs) ;
  
  
  % create main layer
  varargout = cell(1, num_rec_vars) ;
  
  [varargout{:}] = Layer.create(@while_loop, [initial_values, non_rec_vars, ...
    {net, opts.count, opts.concatenate, opts.stopCondition}], ...
    'numInputDer', numel(initial_values) + numel(non_rec_vars)) ;

end


function varargout = while_loop(varargin)
%   Forward:
%   [output1, output2, ...] = while_loop(initial_val1, initial_val2, ...,
%    non_rec_val1, non_rec_val2, ..., net, count, dims, stopCondition)
%
%   Backward:
%   [dinitial_val1, dinitial_val2, ..., dnon_rec_val1, dnon_rec_val2, ...]
%    = while_loop(..., doutput1, doutput2, ...)

  % find Net argument
  pos = [] ;
  for i = 1:numel(varargin)
    if isa(varargin{i}, 'Net')
      pos = i ;
      break ;
    end
  end
  
  % extract arguments
  [net, count, dims, stop] = deal(varargin{pos : pos + 3}) ;
  
  % get output variables' indexes
  output_var_idx = net.meta.output_var ;
  num_outputs = numel(output_var_idx) ;
  
  assert(~stop, 'Not implemented yet.');
  
  % extract remaining arguments
  assert(num_outputs < pos, 'Invalid inputs.') ;
  output_der_ = varargin(pos + 4 : end) ;  % derivatives, if any
  rec_values = varargin(1 : num_outputs) ;  % initial values of recursive vars
  non_rec_vars = varargin(num_outputs + 1 : pos - 1) ;  % non-recursive vars
  
  % get indexes of recursive and non-recursive variables
  rec_var_idx = net.meta.rec_var ;
  non_rec_var_idx = net.meta.non_rec_var ;
  
  % dimensions for output concatenation
  if isscalar(dims)
    dims = dims(ones(1, num_outputs)) ;
  end
  assert(isempty(dims) || numel(dims) == num_outputs) ;
  

  if isempty(output_der_)
    %
    % forward mode
    %
    
    % get index of counter var, or 0 if it doesn't exist
    counter_var_idx = net.meta.counter_var ;

    % set values of non-recursive vars now, used by all iterations
    net.vars(non_rec_var_idx) = non_rec_vars ;

    % allocate activations and outputs for each iteration
    act = cell(1, min(count, 1e4)) ;  % allow inf
    out = cell(numel(act), num_outputs) ;
    
    % clear previous derivatives, so they're not saved with the activations
    net.vars(2:2:end) = {[]} ;
    
    for c = 1:count
      % set values of recursive vars
      net.vars(rec_var_idx) = rec_values ;
      
      % set counter variable
      if counter_var_idx
        net.vars{counter_var_idx} = c ;
      end
      
      % evaluate network
      net.eval({}, 'forward') ;
      
      % save activations
      act{c} = net.vars ;

      % get outputs, which will be fed back in the next iteration
      rec_values = act{c}(output_var_idx) ;
      
      % save outputs
      if ~isempty(dims)
        out(c,:) = rec_values ;
      end
    end
    
    if isempty(dims)
      % only return the outputs of the final iteration
      varargout = rec_values ;
      
    else
      % concatenate outputs for all iterations, over specified dimensions
      varargout = cell(1, num_outputs) ;
      for k = 1:num_outputs
        varargout{k} = cat(dims(k), out{:,k}) ;
        assert(size(varargout{k}, dims(k)) == numel(act), ...
          'The output sizes at each iteration must be 1, for the dimensions specified in DIMS.') ;
      end
    end
    
    % remember activations for backward pass
    net.meta.activations = act ;
    
    
  else
    %
    % backward mode
    %
    
    act = net.meta.activations ;
    assert(num_outputs == numel(output_der_)) ;
    
    % extract output derivatives. note that they may be concatenated
    % tensors of the outputs over all iterations, so break them up.
    if ~isempty(dims)
      output_der = cell(numel(act), num_outputs) ;
      
      max_dims = 1 ;  % create subscripts ':' for all dimensions
      for k = 1:numel(output_der_)
        max_dims = max(max_dims, ndims(output_der_{k})) ;
      end
      subs = cell(1, max_dims) ;
      subs(:) = {':'} ;
      
      for k = 1:numel(output_der_)
        subs_ = subs ;
        o = output_der_{k} ;
        
        if ~isscalar(o)
          for j = 1:size(output_der,1)
            subs_{dims(k)} = j ;  % slice along the specified dim., keeping others ':'
            output_der{j,k} = o(subs_{:}) ;
          end
        else  % edge case, a scalar derivative (e.g. 0)
          output_der(:,k) = {o} ;
        end
      end
    end
    
    for c = numel(act) : -1 : 1
      % get the output derivatives corresponding to this iteration
      if isempty(dims)  % only the final iteration was returned
        if c == numel(act)
          iter_der = output_der_ ;  % final iteration's derivative
        else
          iter_der = rec_der ;  % intermediate derivatives are fed back
        end
      else
        % the outputs of all iterations were returned
        iter_der = output_der(c,:) ;

        if c < numel(act)
          % accumulate them with the derivatives that are fed back from the
          % previous iteration
          for k = 1:num_outputs
            iter_der{k} = iter_der{k} + rec_der{k} ;
          end
        end
      end
      
      % reuse stored activations, which includes the iteration's inputs.
      % note this also clears derivatives.
      net.vars = act{c} ;
      
      % backpropagate through iteration network
      if isscalar(iter_der), iter_der = iter_der{1} ; end  % handle single-element lists
      net.eval({}, 'backward', iter_der) ;
      
      % accumulate derivatives for all non-recursive vars
      der = net.vars(non_rec_var_idx + 1) ;
      
      if c == numel(act)  % first time
        non_rec_der = der ;
      else
        for i = 1:numel(non_rec_vars)
          non_rec_der{i} = non_rec_der{i} + der{i} ;
        end
      end
      
      % get derivatives for recursive vars (to feed back to the previous
      % iteration, or to return them at the end)
      rec_der = net.vars(rec_var_idx + 1) ;
    end
    
    % return input derivatives for recursive vars' initial values (i.e.,
    % the input derivatives of the first iteration), followed by the
    % derivatives for non-recursive vars.
    varargout = [rec_der(:); non_rec_der(:)] ;
  end
end

