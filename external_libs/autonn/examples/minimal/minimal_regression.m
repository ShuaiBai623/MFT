% MINIMAL_REGRESSION
%   Demonstrates automatic differentiation of a least-squares objective.

run('../../setup_autonn.m') ;  % add AutoNN to the path
rng(0) ;  % set random seed


% load simple data (4 features, 3 classes)
s = load('fisheriris.mat') ;
data_x = single(s.meas.') ;  % features-by-samples matrix
[~, ~, data_y] = unique(s.species) ;  % convert strings to class labels



% define inputs and parameters
x = Input() ;
y = Input() ;
w = Param('value', 0.01 * randn(3, 4, 'single')) ;
b = Param('value', 0.01 * randn(3, 1, 'single')) ;

% combine them using math operators, which define the prediction
prediction = w * x + b ;

% compute least-squares loss
loss = sum(sum((prediction - y).^2)) ;

% use workspace variables' names as the layers' names, and compile net
Layer.workspaceNames() ;
net = Net(loss) ;



% simple SGD
learningRate = 1e-5 ;
outputs = zeros(1, 100) ;

for iter = 1:100
  % draw minibatch
  idx = randperm(numel(data_y), 50) ;
  
  % evaluate network to compute gradients
  net.eval({x, data_x(:,idx), y, data_y(idx)'}) ;
  
  % update weights
  net.setValue(w, net.getValue(w) - learningRate * net.getDer(w)) ;
  net.setValue(b, net.getValue(b) - learningRate * net.getDer(b)) ;
  
  % plot loss
  outputs(iter) = net.getValue(loss) ;
end

figure(3) ;
plot(outputs) ;
xlabel('Iteration') ; ylabel('Loss') ;

loss
net
