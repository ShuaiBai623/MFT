classdef nnslice < nntest
  methods (Test)

    function basic(test)
        sz = [3,3,5,4] ;
        x = test.randn(sz) ;
        dim = 4 ;
        slicePoints = 1:dim - 1 ; % slice along fourth dim
        y = vl_nnslice(x, dim, slicePoints, []) ;

        % check derivatives with numerical approximation
        dzdy = cellfun(@(x) test.randn(size(x)), y, 'Uni', 0) ;
        dzdx = vl_nnslice(x, dim, slicePoints, dzdy, 'inputSizes', {sz}) ;
        dzdy_ = cat(dim, dzdy{:}) ;
        dzdx_ = dzdx{1} ;
        test.der(@(x) forward_wrapper(x, dim, slicePoints), x, dzdy_, dzdx_, 1e-3*test.range) ;
    end
  end
end

% -----------------------------------------------------------------
function y = forward_wrapper(x, dim, slicePoints)
% -----------------------------------------------------------------
  y = vl_nnslice(x, dim, slicePoints, []) ;
  y = cat(dim, y{:}) ;
end
