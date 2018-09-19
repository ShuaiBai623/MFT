function generator = fromFunction(func, varargin)
%FROMFUNCTION Generator for new custom layer type
%   GENERATOR = Layer.fromFunction(FUNC) returns a generator for Layer
%   objects with the function FUNC (i.e., a custom layer).
%
%   For example:
%
%      customLoss = Layer.fromFunction(@func) ;
%
%   Now customLoss can be composed with other Layer objects, to define a
%   new node in the computational graph:
%
%      loss = customLoss(prediction, labels) ;
%
%   Compare this to using, for example, the standard vl_nnloss function:
%
%      loss = vl_nnloss(prediction, labels) ;
%
%   Layer.fromFunction enables using new functions as layers in a network.
%
%Interface
%   The interface for a differentiable function FUNC is modeled after the
%   basic blocks of MatConvNet, such as vl_nnconv.
%
%   FUNC must accept extra output derivative arguments, which will be
%   supplied when in backward mode. In the above example, it will be called
%   as:
%   - Forward mode:   Y = FUNC(X, L)
%   - Backward mode:  DZDX = FUNC(X, L, DZDY)
%   Where DZDY is the output derivative and DZDX is the input derivative.
%
%   Alternatively, if a function named FUNC_DER exists in the path, it will
%   be called in backward mode instead of FUNC. This may simplify the
%   definition of your custom layer in some cases.
%
%   If your custom function is called with any name-value pairs, the output
%   derivative arguments will appear *before* the name-value pairs.
%
%
%   GENERATOR = Layer.fromFunction(FUNC, 'option', value, ...) accepts the
%   following options:
%
%   `numInputDer`:: automatic
%     Number of inputs for which a derivative is returned in backward mode.
%     By default, it is assumed that all of them have derivatives. To
%     override this behavior, set numInputDer.
%
%     In the example above, this specifies that there is no derivative for
%     the LABELS input:
%
%       customLoss = Layer.fromFunction(@func, 'numInputDer', 1) ;
%
%     You can define a non-differentiable function with numInputDer = 0,
%     meaning it won't affect the optimization at all. This may be useful
%     to define error metrics or scopes to probe the network's progress.
%
%   `numOutputs`:: automatic
%     Number of outputs of FUNC. This is usually inferred, but can be
%     overriden.

  assert(isa(func, 'function_handle'), 'Argument must be a valid function handle.') ;

  opts = varargin ;
  generator = @(varargin) Layer.create(func, varargin, opts{:}) ;
end

