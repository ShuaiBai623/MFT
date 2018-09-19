% MINIMAL_NETWORK
%   Demonstrates a simple logistic regression network.

run('../../setup_autonn.m') ;  % add AutoNN to the path
rng(0) ;  % set random seed


% load simple data (4 features, 3 classes)
s = load('fisheriris.mat') ;
data_x = single(reshape(s.meas.', 1, 1, 4, [])) ;  % features in 3rd channel
[~, ~, data_y] = unique(s.species) ;  % convert strings to class labels



% define inputs
x = Input() ;
y = Input() ;

% predict using a convolutional layer.
% filter/bias parameters are created and initialized automatically.
% type 'help Layer.vl_nnconv' to see more initialization options.
prediction = vl_nnconv(x, 'size', [1, 1, 4, 3]) ;

% define loss, and classification error
loss = vl_nnloss(prediction, y) ;
error = vl_nnloss(prediction, y, 'loss','classerror') ;

% use workspace variables' names as the layers' names, and compile net
Layer.workspaceNames() ;
net = Net(loss, error) ;


% initialize solver
solver = solvers.SGD() ;
solver.learningRate = 1e-1 ;


errors = zeros(1, 100) ;

for iter = 1:100
  % draw minibatch
  idx = randperm(numel(data_y), 50) ;
  
  % evaluate network to compute gradients
  net.eval({'x', data_x(:,:,:,idx), 'y', data_y(idx)}) ;
  
  % take one SGD step
  solver.step(net) ;
  
  % plot error
  errors(iter) = net.getValue(error) ;
end

figure(3) ;
plot(errors) ;
xlabel('Iteration') ; ylabel('Error') ;

loss
net

