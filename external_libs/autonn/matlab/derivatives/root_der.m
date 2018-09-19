function varargout = root_der(varargin)
%ROOT_DER Derivative of the root layer
%   Simply distributes any output derivative to the inputs. They may be
%   converted to handle scalars and different variable types.
%   See ROOT.

  % copy the output derivative to all input derivatives as-is
  der = cell(1, numel(varargin) - 1) ;
  
  if ~iscell(varargin{end})  % copy same value to all outputs
    der(:) = varargin(end) ;
    
  else  % copy a different value to each output
    assert(numel(varargin{end}) == numel(der), ...
      'Must specify one output derivative for each output of the network (in the derOutput argument of Net.eval).') ;
    der(:) = varargin{end} ;
  end
  
  varargout = cell(size(der)) ;
  
  for i = 1:numel(der)
    % expand scalars, if the corresponding input was a different size.
    % this ensures the size of a var and its derivative are the same.
    % also convert them to the same class (e.g. single, gpuArray).
    % NOTE: don't use cast, as it memory-leaks for gpuArrays (Matlab 2016b)
    
    varargout{i} = zeros(size(varargin{i}), 'like', varargin{i}) ;
    varargout{i}(:) = der{i} ;  % copies array or replicates single element, if scalar
    
  end
end
