function varargout = create(func, args, varargin)
%CREATE Creates a layer from a function handle and arguments
%   OBJ = Layer.create(@FUNC, ARGS) creates OBJ of type Layer that, when
%   evaluated, calls the function FUNC with arguments given in a cell array
%   ARGS. Some elements of ARGS may be Layer objects, which allows
%   composing layers into networks.
%
%   Layer.create(..., 'option', value, ...) accepts additional options.
%   See 'help Layer.fromFunction' for more information.

  assert(isa(func, 'function_handle'), 'Argument must be a valid function handle.') ;

  opts.numInputDer = [] ;
  opts.numOutputs = [] ;
  opts.name = '' ;

  opts = vl_argparse(opts, varargin) ;

  % main output
  varargout = cell(1, nargout) ;
  varargout{1} = Layer(func, args{:}) ;
  varargout{1}.numOutputs = nargout ;  % infer number of layer outputs from this function call
  varargout{1}.numInputDer = opts.numInputDer ;
  varargout{1}.name = opts.name ;

  % selectors for any additional outputs
  for i = 2:nargout
    varargout{i} = Selector(varargout{1}, i) ;
  end
end
