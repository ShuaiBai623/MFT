function obj = extras_autonn_custom_fn(block, inputs, params)
% EXTRAS_AUTONN_CUSTOM_FN autonn custom layer converter for extra
% layers
%
% Copyright (C) 2017 Samuel Albanie
% Licensed under The MIT License [see LICENSE.md for details]

  switch class(block)
    case 'dagnn.Normalize'
      obj = Layer.create(@vl_nnscalenorm, {inputs{1}, params{1}}) ;
    case 'dagnn.Permute'
      obj = Layer.create(@permute, {inputs{1}, block.order}) ;
    case 'dagnn.Flatten'
      obj = Layer.create(@vl_nnflatten, {inputs{1}, block.axis}) ;
    case 'dagnn.GlobalPooling'
      obj = Layer.create(@vl_nnglobalpool, inputs(1)) ;
    case 'dagnn.Reshape'
      obj = Layer.create(@vl_nnreshape, {inputs{1}, block.shape}) ;
    case 'dagnn.Max'
      obj = Layer.create(@vl_nnmax, [{numel(inputs)}, inputs]) ;
    case 'dagnn.Crop'
      obj = vl_nncrop_wrapper(inputs{1}, inputs{2}, block.crop) ;
    case 'dagnn.Axpy'
      obj = Layer.create(@vl_nnaxpy, inputs(1:3)) ;
    case 'dagnn.Interp'
      obj = Layer.create(@vl_nninterp, {inputs{1}, block.shrinkFactor, ...
                        block.zoomFactor, 'padBeg', block.padBeg, ...
                        'padEnd', block.padEnd}) ;
    case 'dagnn.SoftMaxTranspose'
      obj = Layer.create(@vl_nnsoftmaxt, {inputs{1}, 'dim', block.dim}) ;
    case 'dagnn.Scale'
      args = {'numInputDer', 3} ; hasBias = block.hasBias ; % simplify interface
      if hasBias, ins = inputs(1:3) ; else, ins = [inputs(1:2) {[]}] ; end
      obj = Layer.create(@vl_nnscale, [ins {'size', block.size}], args{:}) ;
    case 'dagnn.PriorBox'
      obj = Layer.create(@vl_nnpriorbox, {inputs{1}, inputs{2}, ...
                                  'aspectRatios', block.aspectRatios, ...
                                  'pixelStep', block.pixelStep, ...
                                  'variance', block.variance, ...
                                  'minSize', block.minSize, ...
                                  'maxSize', block.maxSize, ...
                                  'offset', block.offset, ...
                                  'flip', block.flip, ...
                                  'clip', block.clip}, ...
                                  'numInputDer', 0) ;
    case 'dagnn.MultiboxDetector'
      numClasses = double(block.numClasses) ; % fixes weird bug
      obj = Layer.create(@vl_nnmultiboxdetector, ...
                                 {inputs{1}, inputs{2}, inputs{3}...
                                 'numClasses', numClasses, ...
                                 'nmsThresh', block.nmsThresh}, ...
                                 'numInputDer', 0) ;
    case 'SpatialSoftMax'
      obj = Layer.create(@vl_nnspatialsoftmax, inputs) ;
    otherwise, keyboard
  end
