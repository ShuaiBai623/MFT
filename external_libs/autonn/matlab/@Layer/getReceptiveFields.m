function [rfSize, rfOffset, rfStride] = getReceptiveFields(obj, input, customRF)
%GETRECEPTIVEFIELDS Computes the receptive fields of a CNN
%   [RFSIZE, RFOFFSET, RFSTRIDE] = OBJ.GETRECEPTIVEFIELDS()
%   Returns the receptive fields of a single-stream CNN with output OBJ.
%
%   Because receptive fields are assumed to be rectangular, it only makes
%   sense to model them for CNNs (composed of convolutional and
%   element-wise operators only).
%
%   Unknown layers are assumed to be element-wise. It is also assumed that
%   all layers are composed using the first argument only. Unless specified
%   otherwise, the CNN is assumed to start with an Input layer. Taken
%   together these mean that the CNN has a single stream (it is a
%   sequential network). See below on how to override these behaviors.
%
%   All 3 returned values are N-by-2 matrices, where N is the number of
%   layers in the stream. They contain for each layer the size, offset and
%   stride that define a rectangular receptive field.
%
%   They can be interpreted as follows. Given a pixel of vertical
%   coordinate u in an output variable OUT(u,...) , the first and last
%   pixels affecting that pixel in an input variable IN(v,...) are:
%
%     v_first = rfstride(L,1) * (y - 1) + rfoffset(L,1) - rfsize(L,1)/2 + 1
%     v_last  = rfstride(L,1) * (y - 1) + rfoffset(L,1) + rfsize(L,1)/2 + 1
%
%   And likewise (using index (L,2) instead of (L,1)) for the horizontal
%   coordinate. See the MatConvNet Manual (PDF) for a more detailed
%   exposition.
%
%   [...] = OBJ.GETRECEPTIVEFIELDS(INPUT)
%   Defines a different layer as the beginning (input) of the CNN stream.
%
%   [...] = OBJ.GETRECEPTIVEFIELDS(INPUT, CUSTOMRF)
%   Specifies a custom function to handle unknown layers (i.e., define the
%   receptive fields of custom layers). The function signature is:
%
%     [kernelsize, offset, pad, stride] = customRF(obj)
%
%   where obj is the unknown layer, and the returned values are the kernel
%   size, offset, padding and stride respectively. Any empty values will
%   assume their defaults (i.e., an element-wise layer). Note that the
%   offset (translation of the receptive field) can be computed from the
%   padding (zero padding, commonly used with convolution/pooling), so only
%   one must be specified.
%
%   [...] = GETRECEPTIVEFIELDS({LAYER1, LAYER2, ...}, ...)
%   Defines a different sequential stream of layers {LAYER1, LAYER2, ...}.
%   The layers must be in forward order.
%
%   Joao F. Henriques, 2017

  if nargin < 2
    input = [] ;
  end
  if nargin < 3
    customRF = [] ;
  end
  
  % note: could improve error checking here.
  % to do: generalize to whole graph instead of single stream (see
  % DagNN.getVarReceptiveFields).
  if iscell(obj)
    % passed in a sequential list of layers
    layers = obj;
  else
    % traverse backwards on first input (assume single-stream CNN structure)
    layers = {} ;
    while ~isa(obj, 'Input') && ~eq(obj, input, 'sameInstance')
      layers{end+1} = obj ;
      if isa(obj.inputs{1}, 'Layer')
        obj = obj.inputs{1} ;
      else
        break
      end
    end
    
    % change to forward-order
    layers = layers(end:-1:1) ;
  end
  
  
  % allocate memory
  rfSize = zeros(2, numel(layers) + 1);
  rfOffset = rfSize;
  rfStride = rfSize;
  
  % initial receptive field (input)
  totalSz = [1, 1];
  totalOffset = [1, 1];
  totalStride = [1, 1];
  rfSize(:,1) = totalSz ;
  rfOffset(:,1) = totalOffset ;
  rfStride(:,1) = totalStride ;
  
  % compose receptive fields in forward-order
  for i = 1:numel(layers)
    l = layers{i} ;
    
    % find name-value pairs
    a = find(strcmp(l.inputs, 'stride')) ;
    if isempty(a)
      stride(1:2) = 1 ;
    else
      stride(1:2) = l.inputs{a + 1} ;
    end
    a = find(strcmp(l.inputs, 'pad')) ;
    if isempty(a)
      pad(1:4) = 0 ;
    else
      pad(1:4) = l.inputs{a + 1} ;
    end
    a = find(strcmp(l.inputs, 'dilate')) ;
    if isempty(a)
      dilate(1:2) = 1 ;
    else
      dilate(1:2) = l.inputs{a + 1} ;
    end
    
    offset = [] ;
    
    switch func2str(l.func)
    case 'vl_nnconv'
      % convolution
      sz = getKernelSize(l, dilate) ;
      
    case 'vl_nnpool'
      % pooling
      assert(isnumeric(l.inputs{2}), 'Pooling size is not numeric.') ;
      sz(1:2) = l.inputs{2} ;
      
    case 'vl_nnconvt'
      % convolution-transpose
      a = find(strcmp(l.inputs, 'upsample')) ;
      if isempty(a)
        upsample(1:2) = 1 ;
      else
        upsample(1:2) = l.inputs{a + 1} ;
      end
      a = find(strcmp(l.inputs, 'crop')) ;
      if isempty(a)
        crop(1:4) = 0 ;
      else
        crop(1:4) = l.inputs{a + 1} ;
      end
      
      ks = getKernelSize(l, 1) ;
      
      sz = (ks - 1) ./ upsample + 1 ;
      stride = 1 ./ upsample ;
      offset = (2 * crop([1 3]) - ks + 1) ./ (2 * upsample) + 1 ;
      
    otherwise
      % others, assume element-wise by default
      sz(1:2) = 1 ;
      
      if ~isempty(customRF)
        % use handler for custom layers
        [ks, of, pd, st] = customRF(l);  %#ok<RHSFN>
        if ~isempty(ks), sz(1:2) = ks ; end
        if ~isempty(of), offset(1:2) = of ; end
        if ~isempty(pd), pad(1:4) = pd ; end
        if ~isempty(st), stride(1:2) = st ; end
        assert(isempty(of) || isempty(pd), 'Cannot specify padding and offset simultaneously.');
      end
    end
    
    % general case, filter-like operator
    if isempty(offset)
      offset = 1 - pad([1, 3]) + (sz(1:2) - 1) / 2;
    end
    
    % compose receptive fields
    totalSz = totalStride .* (sz - 1) + totalSz ;
    totalOffset = totalStride .* (offset - 1) + totalOffset ;
    totalStride = totalStride .* stride ;
    
    % store results
    rfSize(:,i+1) = totalSz ;
    rfOffset(:,i+1) = totalOffset ;
    rfStride(:,i+1) = totalStride ;
  end
end

function ks = getKernelSize(l, dilate)
  % obtains the kernel size, taking dilation into account.
  % assumes L is a convolution/conv-transpose layer.
  assert(isa(l.inputs{2}, 'Param')) ;
  w = l.inputs{2}.value ;

  ks = max([size(w,1), size(w,2)], 1) ;
  ks = (ks - 1) .* dilate + 1 ;
end

