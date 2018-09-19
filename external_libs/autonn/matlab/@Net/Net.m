classdef Net < handle
%NET Compiled network that can be evaluated on data
%   While Layer objects are used to easily define a network topology
%   (build-time), a Net object compiles them to a format that can be
%   executed quickly (run-time).
%
%   To compile a network, defined by its output Layer objects, just pass
%   them to a Net during construction.
%
%   Example:
%      % define topology
%      images = Input() ;
%      labels = Input() ;
%      prediction = vl_nnconv(images, 'size', [5, 5, 1, 3]) ;
%      loss = vl_nnloss(prediction, labels) ;
%
%      % assign names automatically
%      Layer.workspaceNames() ;
%
%      % compile network
%      net = Net(loss) ;
%
%      % evaluate the network on some input data
%      net.eval({'images', randn(5, 5, 1, 3, 'single'), ...
%                'labels', single(1:3)}) ;
%
%      disp(net.getValue(loss)) ;  % get loss value
%      disp(net.getDer(images)) ;  % get image derivatives
%
%
%   <a href="matlab:properties('Net'),methods('Net')">Properties and methods</a>
%   See also properties('Net'), methods('Net'), Layer.

% Copyright (C) 2016-2017 Joao F. Henriques.
% All rights reserved.
%
% This file is part of the VLFeat library and is made available under
% the terms of the BSD license (see the COPYING file).

  properties (SetAccess = protected, GetAccess = public)
    forward = []  % forward pass function calls
    backward = []  % backward pass function calls
    inputs = struct()  % struct of network's Inputs, indexed by name
    params = []  % list of Params
    gpu = false  % whether the network is in GPU or CPU mode
    isGpuVar = []  % whether each variable or derivative can be on the GPU
    parameterServer = []  % ParameterServer object, accumulates parameter derivatives across GPUs
    conserveMemory = [false, false] % 1x2 logical determining variable deletion on the forward/backward pass
  end
  properties (SetAccess = public, GetAccess = public)
    vars = {}  % cell array of variables and their derivatives (access with getValue/setValue/getDer/setDer)
    meta = []  % optional meta properties
    diagnostics = []  % list of diagnosed vars (see Net.plotDiagnostics)
  end

  methods (Access = private)
    compile(net, varargin)
    optimizeVars(net, opts, objs)
  end
  
  methods
    function net = Net(varargin)
      % NET Constructor for Net object
      %    The constructor accepts a list of Layers (output layers of a
      %    network for compilation), a SimpleNN/DagNN to be converted to
      %    Net, or a saved struct created with SAVEOBJ.
      %
      %    Name-value pairs for compilation are also accepted (see
      %    Net.compile).
      
      if nargin == 0, return, end  % empty constructor
      
      % separate name-value pairs (compilation args) from object inputs
      firstString = find(cellfun(@ischar, varargin), 1) ;
      if isempty(firstString)  % no name-value pairs
        objects = varargin ;
        args = {} ;
      else  % separate them
        objects = varargin(1:firstString-1) ;
        args = varargin(firstString:end) ;
      end
      
      % load from struct, distinguishing from SimpleNN
      if isscalar(objects) && isstruct(objects{1}) && ~isfield(objects{1}, 'layers')
        net = Net.loadobj(objects{1}) ;
        assert(isempty(args), ...
          'Compilation arguments cannot be used when just loading a pre-compiled Net.') ;
        return
      end

      % if not a set of layers, assume SimpleNN or DagNN and convert first
      if isscalar(objects) && ~isa(objects{1}, 'Layer')
        objects = Layer.fromDagNN(objects{:}) ;
        
        if ~iscell(objects)
          objects = {objects} ;  % objects should contain a list of Layer objects
        end
      end

      % compile Net from a list of Layers
      net.compile(objects{:}, args{:}) ;
    end
    
    function value = getValue(net, var)
      %GETVALUE Returns the value of a given variable
      %   OBJ.GETVALUE(VAR) returns the value of a given variable.
      %   VAR may be a Layer object, its name, or an internal var index.
      [idx, isList] = net.getVarIndex(var) ;
      if ~isList
        value = net.vars{idx} ;
      else
        value = net.vars(idx) ;
      end
    end
    
    
    function der = getDer(net, var)
      %GETDER Returns the derivative of a given variable
      %   OBJ.GETDER(VAR) returns the derivative of a given variable.
      %   VAR may be a Layer object, its name, or an internal var index.
      [idx, isList] = net.getVarIndex(var) ;
      if ~isList
        der = net.vars{idx + 1} ;
      else
        der = net.vars(idx + 1) ;
      end
    end
    
    function setValue(net, var, value)
      %SETVALUE Sets the value of a given variable
      %   OBJ.SETVALUE(VAR, VALUE) sets the value of a given variable.
      %   VAR may be a Layer object, its name, or an internal var index.
      [idx, isList] = net.getVarIndex(var) ;
      if ~isList
        net.vars{idx} = value ;
      else
        net.vars(idx) = value ;
      end
    end
    
    function setDer(net, var, der)
      %SETDER Sets the derivative of a given variable
      %   OBJ.SETDER(VAR) sets the derivative of a given variable.
      %   VAR may be a Layer object, its name, or an internal var index.
      %
      %   Note that the network output derivatives are set in the call to
      %   Net.eval, and the others are computed with backpropagation, so
      %   there is rarely a need to call this function.
      [idx, isList] = net.getVarIndex(var) ;
      if ~isList
        net.vars{idx + 1} = der ;
      else
        net.vars(idx + 1) = der ;
      end
    end
    
    function [idx, isList] = getVarIndex(net, var, errorIfNotFound)
      %GETVARINDEX Returns the internal index of a variable
      %   OBJ.GETVARINDEX(VAR) returns the index IDX of OBJ.VARS{IDX},
      %   where the variable is stored. VAR may be a Layer object or its
      %   name. This may be used to speed up some operations, but is mostly
      %   intended for internal use. The corresponding derivative is stored
      %   in OBJ.VARS{IDX + 1}.
      %
      %   OBJ.GETVARINDEX(VAR, TRUE) returns 0 if the variable is not
      %   found, instead of throwing an error.
      if nargin < 3
        errorIfNotFound = true ;
      end
      isList = false ;  % to deal with edge-case of setValue({var}, {value}) (single-elem. lists)
      if ischar(var)
        % search for var/layer by name
        if isfield(net.inputs, var)  % search inputs
          idx = net.inputs.(var) ;
        else  % search params
          param = strcmp({net.params.name}, var) ;
          if any(param)
            idx = net.params(param).var ;
          else  % search layers
            layer = strcmp({net.forward.name}, var) ;
            if any(layer)
              idx = net.forward(layer).outputVar ;
            else
              if errorIfNotFound,
                error(['No var with specified name ''' var '''.']) ;
              end
              idx = 0 ;
            end
          end
        end
      elseif isa(var, 'Layer')
        idx = var.outputVar(1) ;
      elseif iscell(var)
        idx = zeros(size(var)) ;
        for i = 1:numel(idx)
          idx(i) = net.getVarIndex(var{i}, errorIfNotFound) ;
        end
        isList = true ;
      else
        assert(isnumeric(var), 'VAR must either be a layer name, a Layer object, or var indexes.') ;
        idx = var ;
        isList = ~isscalar(idx) ;
      end
    end
    
    function useGpu(net, index)
    %MOVE Move data to CPU or GPU
    %   OBJ.USEGPU(INDEX) enables GPU mode, using the GPU with a given
    %   index, or CPU mode, in case the INDEX is empty.
    %
    %   This is done by calling gpuDevice(INDEX), and converting to
    %   gpuArrays all variables marked as such. Inputs and Params are
    %   marked by setting their 'gpu' property to true (Inputs default to
    %   false, Params to true).
    %
    %   The gpuDevice call can be skipped by setting INDEX to 0.
    
      if ~isempty(index) && index ~= 0
        gpuDevice(index) ;
      end
      
      [net.vars, net.isGpuVar] = Net.moveVars(net.vars, net.isGpuVar, ~isempty(index)) ;
      net.gpu = ~isempty(index) ;
      if isfield(net.inputs, 'gpuMode')
        net.setValue('gpuMode', net.gpu) ;
      end
    end
    
    function move(net, device)
    %MOVE Move data to CPU or GPU
    %   OBJ.MOVE(DESTINATION) moves variables to the 'gpu' or the 'cpu'.
    %   Note that the use of Net.useGpu is preferred.
      switch device
      case 'cpu', net.useGpu([]) ;
      case 'gpu', net.useGpu(0) ;
      otherwise, error('Must specify ''gpu'' or ''cpu''.') ;
      end
    end
    
    function clearParameterServer(net)
    %CLEARPARAMETERSERVER Remove the parameter server
    %   OBJ.CLEARPARAMETERSERVER() stops using the parameter server, for
    %   multi-GPU networks. See ParameterServer.
      if ~isempty(net.parameterServer)
        net.parameterServer.stop() ;
      end
      net.parameterServer = [] ;
    end
    
    function reset(net)
    %RESET Alias for clearParameterServer
      net.clearParameterServer();
    end
    
    function display(net, name)
    %DISPLAY Displays network information
      if nargin < 2
        name = inputname(1) ;
      end
      fprintf('\n%s = Net object with:\n\n', name) 
      
      s.Number_of_layers = numel(net.forward) ;
      s.Number_of_variables = numel(net.isGpuVar) ;
      s.Number_of_inputs = numel(fieldnames(net.inputs)) ;
      s.Number_of_parameters = numel(net.params) ;
      s.GPU_mode = net.gpu ;
      s.Multiple_GPUs = ~isempty(net.parameterServer) ;
      
      fprintf(strrep(evalc('disp(s)'), '_', ' ')) ;
      
      showLinks = ~isempty(name) && usejava('desktop') ;
      
      if showLinks
        props = ['<a href="matlab:disp(' name ')">show all properties</a>'] ;
      else
        props = 'use net.disp() to show all properties' ;
      end
      
      if ~isempty(net.vars)
        if showLinks
          fprintf('<a href="matlab:%s.displayVars()">Display variables</a>, %s\n\n', name, props) ;
        else
          fprintf('Use net.displayVars() to show all variables, %s.\n\n', props) ;
        end
      else
        if showLinks
          fprintf('<a href="matlab:%s.displayVars(vars)">Display variables</a>, %s\n', name, props) ;
        else
          fprintf('Use net.displayVars(vars) to show all variables, %s\n', props) ;
        end
        fprintf(['NOTE: Net.eval() is executing. For performance, it holds all of the\n' ...
                 'network''s variables in a local variable (called ''vars''). To display\n' ...
                 'them, first navigate to the scope of Net.eval() with dbup/dbdown.\n\n']) ;
      end
      
    end
    
    function s = saveobj(net)
    %SAVEOBJ Returns the object as a struct
      s.forward = net.forward ;
      s.backward = net.backward ;
      s.inputs = net.inputs ;
      s.params = net.params ;
      s.conserveMemory = net.conserveMemory ;
      s.meta = net.meta ;
      s.diagnostics = net.diagnostics ;
      
      % only save var contents corresponding to parameters, all other vars
      % are transient. also move them from GPU to CPU, and update GPU-ness.
      s.vars = cell(size(net.vars)) ;
      s.isGpuVar = net.isGpuVar ;
      idx = [net.params.var] ;
      [s.vars(idx), s.isGpuVar(idx)] = Net.moveVars( ...
        net.vars(idx), net.isGpuVar(idx), false) ;
    end
  end
  
  methods (Static)
    function net = loadobj(s)
    %LOADOBJ Loads the object from a struct (called by constructor)
      net = Net() ;
      net.forward = s.forward ;
      net.backward = s.backward ;
      net.vars = s.vars ;
      net.inputs = s.inputs ;
      net.params = s.params ;
      net.conserveMemory = s.conserveMemory ;
      net.gpu = false ;
      net.isGpuVar = s.isGpuVar ;
      net.meta = s.meta ;
      net.diagnostics = s.diagnostics ;
    end
  end
  
  methods (Static, Access = private)
    function layer = parseArgs(layer, args)
    %PARSEARGS
    %   Helper function to parse a layer's arguments, storing the constant
    %   arguments (args), non-constant var indexes (inputVars), and their
    %   positions in the arguments list (inputArgPos).
      inputVars = [] ;
      inputArgPos = [] ;
      for a = 1:numel(args)
        if isa(args{a}, 'Layer')
          % note only the first output is taken if there's more than one;
          % other outputs are reached using Selectors
          inputVars(end+1) = args{a}.outputVar(1) ;  %#ok<*AGROW>
          inputArgPos(end+1) = a ;
          args{a} = [] ;
        end
      end
      layer.args = args ;
      layer.inputVars = inputVars ;
      layer.inputArgPos = inputArgPos ;
      layer = orderfields(layer) ;  % have a consistent field order, to not botch assignments
    end
    
    function s = initStruct(n, varargin)
    %INITSTRUCT
    %   Helper function to initialize a struct with given fields and size.
    %   Note fields are sorted in ASCII order (important when assigning
    %   structs).
      varargin(2,:) = {cell(1, n)} ;
      s = orderfields(struct(varargin{:})) ;
    end
    
    function [vars, isGpuVar] = moveVars(vars, isGpuVar, toGpu)
    %MOVEVARS Move data to CPU/GPU, given as arrays (used by SAVEOBJ/MOVE)
      if toGpu
        % only move vars marked as GPU arrays
        vars(isGpuVar) = cellfun(@gpuArray, vars(isGpuVar), 'UniformOutput',false) ;
      else
         % by moving to the CPU we lose the knowledge of which vars are
         % supposed to be on the GPU, so store that. once on the GPU,
         % always on the GPU.
        isGpuVar = isGpuVar | cellfun('isclass', vars, 'gpuArray') ;

        % move all just to be safe
        vars = cellfun(@gather, vars, 'UniformOutput',false) ;
      end
    end
  end
end

