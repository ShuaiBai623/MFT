function y = vl_nnspatialsoftmax(x, varargin)
% NOTE: untested

  [~, dzdy] = vl_argparsepos(struct(), varargin) ;

  if isempty(dzdy)
    reshapein = reshape(x, 1,1,[],size(x,3)*size(x,4)) ;
    y = vl_nnsoftmax(reshapein) ;
    y = reshape(y, size(x)) ;
  else
    reshapein = reshape(x, 1,1,[],size(x,3)*size(x,4)) ;
    reshapedout = reshape(dzdy{1}, 1,1,[],size(x,3)*size(x,4)) ;
    y = vl_nnsoftmax(reshapein, reshapedout) ;
    y = reshape(y, size(x)) ;
  end
