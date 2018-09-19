% MINIMAL_LSTM
%   Demonstrates LSTMs on a toy binary addition problem.
%   The task is to predict the sequence of bits of the sum of two numbers,
%   given their binary sequences (one bit at a time).

run('../../setup_autonn.m') ;  % add AutoNN to the path
rng(0) ;  % set random seed


T = 8 ;  % number of bits / time steps
d = 16 ;  % dimensionality of the LSTM state
iters = 1500 ;  % number of iterations


% inputs
x = Input() ;
y = Input() ;

% initialize the shared parameters for an LSTM with d hidden units and
% two inputs
[W, b] = vl_nnlstm_params(d, 2) ;

% initial state
h = cell(T+1, 1);
c = cell(T+1, 1);
h{1} = zeros(d, 1, 'single');
c{1} = zeros(d, 1, 'single');

% run LSTM over all time steps
for t = 1:T
  [h{t+1}, c{t+1}] = vl_nnlstm(x(:,t), h{t}, c{t}, W, b);
end

% concatenate output into m x T matrix, ignoring initial state (h{1})
H = [h{2:end}] ;

% final projection (note the same projection is applied at all time steps)
prediction = vl_nnconv(reshape(H, 1, 1, d, T), 'size', [1, 1, d, 1]) ;


% define loss, and classification error
loss = vl_nnloss(prediction, y, 'loss', 'logistic') ;
err = vl_nnloss(prediction, y, 'loss','binaryerror') ;


% use workspace variables' names as the layers' names, and compile net
Layer.workspaceNames() ;
net = Net(loss, err) ;


% initialize solver
solver = solvers.Adam() ;
solver.learningRate = 1e-2 ;


losses = zeros(1, iters) ;
errors = zeros(1, iters) ;

for iter = 1:iters
  % generate a simple binary addition problem
  A = randi([0, 2^(T-1) - 1]) ;  % last bit is always 0 to prevent overflow
  B = randi([0, 2^(T-1) - 1]) ;
  C = A + B ;  % true answer
  
  % convert to vectors of binary digits
  data_x = [dec2bin(A, T); dec2bin(B, T)] ;  % concatenate binary numbers
  data_x = single(fliplr(data_x) == '1') ;  % convert from string to double, and reverse sequence
  data_y = dec2bin(C, T) ;  % same for the true answer
  data_y = single(fliplr(data_y) == '1') * 2 - 1;  % bit classes for prediction will be -1 or 1
  
  
  % evaluate network to compute gradients
  net.eval({'x', data_x, 'y', data_y}) ;
  
  % take one SGD step
  solver.step(net) ;
  
  
  % plot loss and error
  losses(iter) = net.getValue(loss) ;
  errors(iter) = net.getValue(err) ;
  
  if mod(iter, 100) == 0
    % display the current prediction
    fprintf('Iteration %i\n', iter);
    fprintf('True sequence: %s\n', sprintf('%i ', data_y > 0));
    fprintf('Predicted:     %s\n', sprintf('%i ', net.getValue('prediction') > 0));
    fprintf('Errors: %i\n\n', errors(iter) * T);
  end
end

figure(3) ;
subplot(1, 2, 1) ; plot(1 : 10 : iters, losses(1 : 10 : iters)) ;
xlabel('Iteration') ; ylabel('Loss') ;
subplot(1, 2, 2) ; plot(1 : 10 : iters, errors(1 : 10 : iters)) ;
xlabel('Iteration') ; ylabel('Error') ;

loss
net

