function layer = vl_nnconv(varargin)
%VL_NNCONV Additional options for vl_nnconv (CNN convolution)
%   Y = Layer.vl_nnconv(X, F, B) computes the convolution of the image X
%   with the filter bank F and biases B. See help vl_nnconv for more
%   details.
%
%   This method overloads MatConvNet's vl_nnconv function for Layer
%   objects, so that instead of executing vl_nnconv, a new Layer object is
%   returned. X, F and B can be other Layers, including Params, or
%   constants.
%
%   Y = Layer.vl_nnconv(X, 'size', SZ) automatically creates parameters F
%   and B, initialized with random filters of size SZ and zero bias. The
%   filters are normally-distributed, with Xavier scaling by default.
%
%   In addition to those defined by MatConvNet's vl_nnconv, the overloaded
%   VL_NNCONV(..., 'option', value, ...) accepts the following options:
%
%   `hasBias`:: true
%     Allows disabling the creation of the bias Param in the syntax above.
%
%   `weightScale`:: 'xavier'
%     If set to a real number, scales the filters by that number instead of
%     using Xavier initialization.
%
%   `learningRate`:: 1
%     Factor used to adjust the created Params' learning rate. Can specify
%     separate learning rates for F and B with a 2-elements vector.
%
%   `weightDecay`:: [1 0]
%     Factor used to adjust the created Params' weight decay. Can specify
%     separate weight decays for F and B with a 2-elements vector.
%
%   `pad`:: [0 0 0 0]
%     Specifying 'same' for this option will set the padding amount such
%     that the input and output spatial dimensions will be the same.
%     This can only be used in conjunction with the 'size' option or if F
%     is of class Param (i.e., the filter size can be inferred).
%
%   `transpose`:: false
%     If true, creates a vl_nnconvt (convolution-transpose) layer instead.

% Copyright (C) 2016-2017 Joao F. Henriques.
% All rights reserved.
%
% This file is part of the VLFeat library and is made available under
% the terms of the BSD license (see the COPYING file).

  % parse options. other options such as 'stride' will be maintained in
  % convArgs.
  opts = struct('size', [], 'weightScale', 'xavier', 'hasBias', true, ...
    'learningRate', [1 1], 'weightDecay', [1 0], 'pad', 0, 'transpose', false) ;
  [opts, posArgs, convOpts] = vl_argparsepos(opts, varargin, 'flags', ...
    {'cuDNN', 'noCuDNN', 'verbose', 'noDerData', 'noDerFilters', 'noDerBiases'}) ;
  
  if ~isempty(opts.size)
    % a size was specified, create Params
    assert(numel(posArgs) == 1, ...
      'Must specify only one input Layer when using the ''size'' option.') ;
    
    if opts.hasBias
      % create bias as the 3rd input
      if opts.transpose  % vl_nnconvt, use 3rd dimension of filters
        biasSize = opts.size(3) ;
      else  % vl_nnconv, use 4th dimension of filters
        biasSize = opts.size(4) ;
      end
      posArgs{3} = Param('value', zeros(biasSize, 1, 'single'), ...
                     'learningRate', opts.learningRate(max(1,end)), ...
                     'weightDecay', opts.weightDecay(max(1,end))) ;
    else
      posArgs{3} = [] ;
    end

    if isequal(opts.weightScale, 'xavier')
      scale = sqrt(2 / prod(opts.size(1:3))) ;
    else
      scale = opts.weightScale ;
    end

    % create filters as the 2nd input
    posArgs{2} = Param('value', randn(opts.size, 'single') * scale, ...
                    'learningRate', opts.learningRate(1), ...
                    'weightDecay', opts.weightDecay(1)) ;
  else
    assert(numel(posArgs) == 3, ...
      'Must specify all 3 inputs, or the ''size'' option.') ;
  end
  
  % handle 'same' padding (i.e., maintain same output size, given stride 1)
  if isequal(opts.pad, 'same')
    assert(isa(posArgs{2}, 'Param'), ...
      'Can only use ''same'' padding when the convolution filter is of class Param.') ;
    sz = size(posArgs{2}.value) ;
    
    pad = (sz(1:2) - 1) / 2 ;  % will be fractional for even filter sizes
    opts.pad = [floor(pad(1)), ceil(pad(1)), floor(pad(2)), ceil(pad(2))] ;
  end
  
  % don't include the padding argument if it's 0
  if any(opts.pad ~= 0)
    convOpts(end+1:end+2) = {'pad', opts.pad} ;
  end
  
  % create layer
  if opts.transpose
    func = @vl_nnconvt ;
  else
    func = @vl_nnconv ;
  end
  layer = Layer(func, posArgs{:}, convOpts{:}) ;
end
