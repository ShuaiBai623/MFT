function y = vl_nncaffepool(x, poolSize, varargin)
%VL_NNCAFFEPOOL(X) - Apply Caffe-style padded pooling
%    Y = VL_NNCAFFEPOOL(X, POOLSIZE) This layer implements Caffe-style pooling,
%    which applies the pooling kernel to the padding, as well as to the input
%    (unlike MatConvNet which does not include the padding itself in the
%    kernel computation). To mimic this effect, we "pre-pad" the input using
%    an empty convolution, then pass to the standard MatConvNet pooling layer.
%    See the docs fo VL_NNPOOL to understand the full pooling operation.
%
%   VL_NNCAFFEPOOL(..., 'option', value, ...) takes the following option:
%
%   `Method`:: 'max'
%     Specify method of pooling. It can be either 'max' (retain max value
%     over the pooling region per channel) or 'avg' (compute the average
%     value over the pooling region per channel).
%
%   `Pad`:: 0
%     The amount of input padding. Input images are padded with zeros
%     by this number of pixels on all sides before the convolution is
%     computed. It can also be a vector [TOP BOTTOM LEFT RIGHT] to
%     specify a different amount of padding in each direction. The
%     size of the pooling filter has to exceed the padding. Unlike standard
%     VL_NNPOOL, this padding is added before the pooling kernel is applied.
%
% Copyright (C) 2018 Samuel Albanie
% Licensed under The MIT License [see LICENSE.md for details]

  opts.pad = 0 ;
  opts.stride = 1 ;
  opts.method = 'avg' ;
  opts.extraArgs = {'cuDNN'} ;
  [opts, dzdy] = vl_argparsepos(opts, varargin) ;

  padded = vl_nnconv(x, [], [], 'pad', opts.pad, opts.extraArgs{:}) ;

  if isempty(dzdy)
    y = vl_nnpool(padded, poolSize, 'pad', 0, 'stride', opts.stride, ...
                  'method', opts.method, opts.extraArgs{:}) ;
  else
    dzdy1 = vl_nnpool(padded, poolSize, dzdy{1}, 'pad', 0, 'stride', opts.stride, ...
                  'method', opts.method, opts.extraArgs{:}) ;
    dzdy = vl_nnconv(x, [], [], dzdy1, 'pad', opts.pad, opts.extraArgs{:}) ;
    y = dzdy ;
  end
