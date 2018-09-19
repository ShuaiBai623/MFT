classdef nndecompile < nntest
  methods (Test)
    function testDiamond(test)
      % diamond topology
      a = Input() ;
      b = sqrt(a) ;
      c = abs(a) ;
      d = b + c ;
      e = exp(d) ;
      
      test.do(e) ;
    end
    
    function testLeNet(test)
      % LeNet + batchnorm + loss
      images = Input('name', 'images', 'gpu', true) ;
      labels = Input('name', 'labels') ;
      
      x = vl_nnconv(images, 'size', [5, 5, 1, 20], 'weightScale', 0.01) ;
      x = vl_nnbnorm(x) ;
      x = vl_nnpool(x, 2, 'stride', 2) ;
      x = vl_nnconv(x, 'size', [5, 5, 20, 50], 'weightScale', 0.01) ;
      x = vl_nnbnorm(x) ;
      x = vl_nnpool(x, 2, 'stride', 2) ;
      x = vl_nnconv(x, 'size', [4, 4, 50, 500], 'weightScale', 0.01) ;
      x = vl_nnbnorm(x) ;
      x = vl_nnrelu(x) ;
      x = vl_nnconv(x, 'size', [1, 1, 500, 10], 'weightScale', 0.01) ;
      
      loss = vl_nnloss(x, labels) ;
      
      test.do(loss) ;
    end
  end
  
  methods
    function do(test, layer)
      % compile
      net1 = Net(layer) ;
      
      % decompile
      decompiledLayers = Layer.fromCompiledNet(net1) ;
      
      % compile again
      net2 = Net(decompiledLayers{:}) ;
      
      % check equality, as structs. erase source code info since it will
      % differ.
      net1 = net1.saveobj() ;
      net2 = net2.saveobj() ;
      
      [net1.forward.source] = deal([]) ;
      [net1.backward.source] = deal([]) ;
      [net1.params.source] = deal([]) ;
      
      [net2.forward.source] = deal([]) ;
      [net2.backward.source] = deal([]) ;
      [net2.params.source] = deal([]) ;
      
      % note: for debugging, doing a "diff" of the two structs is helpful.
      % use structeq: http://mathworks.com/matlabcentral/fileexchange/27542
      % [iseq, info] = structeq(net1, net2, true, true)
      assert(isequal(net1, net2)) ;
    end
  end
end
