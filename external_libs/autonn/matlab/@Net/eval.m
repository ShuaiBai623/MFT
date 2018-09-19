function eval(net, inputs, mode, derOutput, accumulateParamDers)
%EVAL Network evaluation, including backpropagation to compute derivatives
%   OBJ.EVAL(INPUTS) evaluates a network on some inputs, running both
%   forward and backpropagation. INPUTS = {'input1', value1, 'input2',
%   value2, ...} is a cell array with input layer names (layers of class
%   Input) and their values (for example, images and labels).
%
%   After calling EVAL, the values and derivatives can be retrieved with
%   OBJ.GETVALUE and OBJ.GETDER.
%
%   INPUTS can also contain Input layer objects instead of their names.
%
%   OBJ.EVAL(INPUTS, MODE) allows specifying one of the mode strings:
%     * 'normal' performs forward and backpropagation (the default).
%     * 'test' performs only the forward computation (no derivatives), and
%       also sets the testMode input to true, which can be used by layers
%       to behave differently in test mode (e.g. bypassing dropout,
%       freezing batch-normalization).
%     * 'forward' performs only the forward computation (no derivatives),
%       without setting testMode to true.
%     * 'backward' performs only the backward computation (assumes eval was
%       ran in forward mode beforehand).
%
%   OBJ.EVAL(INPUTS, MODE, DEROUTPUT) also specifies the output derivatives
%   of the network. This can be a single value, or multiple (one per
%   network output). Scalars are expanded to full tensors if needed. The
%   default is DEROUTPUT = 1.
%
%   OBJ.EVAL(INPUTS, MODE, DEROUTPUT, false) prevents the network from
%   clearing the derivatives of parameters before backpropagation. This
%   means that parameter derivatives will be accumulated from a previous
%   run. This behavior may be useful to implement sub-batches.

% Copyright (C) 2016 Joao F. Henriques.
% All rights reserved.
%
% This file is part of the VLFeat library and is made available under
% the terms of the BSD license (see the COPYING file).

  if nargin < 2 || ~iscell(inputs) || mod(numel(inputs), 2) ~= 0
    error('Expected network inputs in the form {INPUT1, VALUE1, INPUT2, VALUE2,...}.') ;
  end
  if nargin < 3
    mode = 'normal' ;
  end
  if nargin < 4
    derOutput = single(1) ;
  end
  if nargin < 5
    accumulateParamDers = false ;
  end
  
  % set inputs
  for i = 1 : 2 : numel(inputs) - 1
    var = net.getVarIndex(inputs{i}) ;
    value = inputs{i+1} ;
    
    if net.gpu && net.isGpuVar(var) && ~isa(value, 'gpuArray')
      value = gpuArray(value) ;  % move to GPU if needed
    end
    
    net.vars{var} = value ;
  end

  switch mode
  case {'normal', 'forward', 'backward'}
    testMode = false ;
  case 'test'  % test mode
    testMode = true ;
  otherwise
    error('Unknown mode ''%s''.', mode) ;
  end
  if isfield(net.inputs, 'testMode')
    net.vars{net.inputs.testMode} = testMode ;
  end
  
  
  % use local variables for efficiency
  forward = net.forward ;
  vars = net.vars ;
  conserveMemoryForward = net.conserveMemory(1);
  conserveMemoryBackward = net.conserveMemory(2);
  net.vars = {} ;  % allows Matlab to release memory when needed

  % forward pass
  if ~strcmp(mode, 'backward')
    for k = 1:numel(forward)
      layer = forward(k) ;
      args = layer.args ;
      args(layer.inputArgPos) = vars(layer.inputVars) ;
      out = cell(1, max(layer.outputArgPos)) ;
      if isfield(layer,'debugStop') && layer.debugStop
        fprintf('debug stop at layer %s ...\n',layer.name);
        keyboard
      end
      [out{:}] = layer.func(args{:}) ;
      vars(layer.outputVar) = out(layer.outputArgPos);
      
      % delete non precious variables not needed for backward pass
      if conserveMemoryForward && numel(layer.deleteVars)
        for i = 1:numel(layer.deleteVars)
          % save size and type proxy struct for layers like reshape
          v = vars{layer.deleteVars(i)} ;
         	vars{layer.deleteVars(i)} = struct('size',size(v),...
            'type', cast(0,'like',v)) ;
        end
      end
    end
  end

  % backward pass
  if strcmp(mode, 'normal') || strcmp(mode, 'backward')
    % clear all derivatives. derivatives are even-numbered vars.
    clear = repmat([false; true], numel(vars) / 2, 1);
    if accumulateParamDers  % except for params (e.g. to implement sub-batches)
      clear([net.params.var] + 1) = false ;  % next var is the derivative
    end
    vars(clear) = {0} ;

    % set root layer's output derivative
    assert(~isempty(derOutput), 'Must specify non-empty output derivatives for normal mode.')
    vars{end} = derOutput ;

    backward = net.backward ;

    for k = 1:numel(backward)
      % populate function arguments with input vars and derivatives
      layer = backward(k) ;
      args = layer.args ;
      inputArgPos = layer.inputArgPos ;
      args(inputArgPos) = vars(layer.inputVars) ;

      if ~isequal(layer.func, @slice_wrapper)
        % call function and collect outputs
        out = cell(1, layer.numInputDer) ;
        [out{:}] = layer.func(args{:}) ;

        % sum derivatives. the derivative var corresponding to each
        % input comes right next to it in the vars list. note that some
        % outputs may be ignored (because they're not input layers,
        % just constant arguments).
        inputDers = layer.inputVars + 1 ;  % note this includes incorrect indexes at output der args, but they'll be ignored with FIND
        if layer.accumDer
          for i = find(inputArgPos <= numel(out))
            vars{inputDers(i)} = vars{inputDers(i)} + out{inputArgPos(i)} ;
          end
        else
          % special case, do not accumulate derivatives; used to implement
          % ReLU short-circuiting.
          ii = inputArgPos <= numel(out) ;
          vars(inputDers(ii)) = out(inputArgPos(ii)) ;
        end
      else
        % special case, indexing. the derivative update might be sparse.
        % args = {X, I1, I2, ..., DYDZ}, derivative of X(I1, I2, ...).
        inputDer = layer.inputVars(1) + 1 ;  % index of input derivative var
        subs = args(2:end-1) ;  % indexing subscripts
        
        % there's a fast sparse update that doesn't handle repeated
        % indexes, and a slow one that does. to do: MEX file.
        repeats = false ;  % check for repeated indexes
        for i = 1:numel(subs)
          if ~ischar(subs{i}) && any(diff(sort(subs{i}(:))) == 0)  % faster than unique()
            repeats = true ;
            break
          end
        end
        if ~repeats
          % very efficient, but doesn't handle repeated indexes
          if isequal(vars{inputDer}, 0)  % must initialize with the right size and class
            vars{inputDer} = zeros(size(vars{inputDer - 1}), 'like', args{end}) ;
          end
          vars{inputDer}(subs{:}) = vars{inputDer}(subs{:}) + args{end} ;
        else
          % fall back to dense update if needed
          vars{inputDer} = vars{inputDer} + slice_der(args{:}) ;
        end
      end
      
      % remove derivatives that are no longer needed
      if conserveMemoryBackward
        vars(layer.deleteVars) = {[]};
      end
    end
  end


  % send parameter derivatives to the parameter server, if using multiple
  % GPUs
  if ~isempty(net.parameterServer)
    ps = net.parameterServer ;
    paramDerIdx = [net.params.var] + 1 ;  % indexes of param derivative vars
    paramDer = vars(paramDerIdx) ;
    
    for i = 1:numel(net.params)  % push each one into the parameter server
      ps.pushWithIndex(i, paramDer{i}) ;
    end
    
    vars(paramDerIdx) = {[]} ;  % clear them
  end
  
  net.vars = vars ;
end

