classdef nncheckder < nntest
  methods
    function y = checkDer(test, output, conserveMemory, ignore, step, tol)
      % checks derivatives of a network (output) w.r.t. all Param layers.
      % also returns computed output (may be useful for additional checks).
      
      if nargin < 4, ignore = [] ; end
      if nargin < 5, step = 1e-6 * test.range ; end
      if nargin < 6, tol = [] ; end
      
      % compile net
      net = Net(output, 'conserveMemory', conserveMemory) ;
      
      % run forward only
      net.eval({}, 'forward') ;
      
      % check output is non-empty
      y = net.getValue(output) ;
      test.verifyNotEmpty(y) ;
      
      % create derivative with same size as output
      der = randn(size(net.getValue(output)), test.currentDataType) ;
      
      % handle GPU
      if strcmp(test.currentDevice, 'gpu')
        gpuDevice(1) ;
        net.move('gpu') ;
        der = gpuArray(der) ;
      end
      
      % run forward and backward
      net.eval({}, 'normal', der) ;
      
      % check derivatives w.r.t. any Param layers
      params = output.find('Param') ;
      values = cellfun(@(p) {net.getValue(p)}, params) ;
      
      guessTol = isempty(tol) ;
      
      for p = 1:numel(params)
        if eq(params{p}, ignore, 'sameInstance'), continue, end
        
        test.verifyNotEmpty(net.getDer(params{p})) ;
        
        wrapper = @(x) forward_wrapper(net, params, values, p, x, output) ;
        
        % refresh
        net.setValue(params, values);
        net.eval({}, 'normal', der) ;
        dzdx = net.getDer(params{p}) ;
        
        if guessTol
          % set numerical check tolerance based on derivative magnitudes
          maxv = max(real(dzdx(:))) ;
          minv = min(real(dzdx(:))) ;
          tol = max(1e-2 * (maxv - minv), 1e-3 * max(maxv, -minv)) ;
        end
        
        test.der(wrapper, values{p}, der, dzdx, step, tol) ;
      end
    end
  end
end

function res = forward_wrapper(net, params, values, pos, x, output)
  % update current variable (for numerical derivative)
  values{pos} = x ;
  
  % assign values to all Params
  net.setValue(params, values) ;

  net.eval({}, 'forward') ;
  res = net.getValue(output) ;
end

