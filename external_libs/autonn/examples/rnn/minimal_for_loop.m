% MINIMAL_FOR_LOOP
%   Simple demonstration of dynamic, fully differentiable For-loops.

run('../../setup_autonn.m') ;  % add AutoNN to the path


% network inputs
num_iterations = Input();  % number of For iterations
initial_x = Input();  % initial value

% this defines an iteration that simply doubles a value at each step.
% note the iteration counter (t) is always passed in as the last argument.
iteration = @(x, t) 2 * x;

% create For-loop (this could be fed back into other parts of the graph)
y = For(iteration, initial_x, num_iterations) ;


% name layers automatically, and compile network
Layer.workspaceNames();
net = Net(y);

net


% run network with 2 For-loop iterations
net.eval({'initial_x', 1, 'num_iterations', 2});

fprintf('Output y after 2 iterations = %g\n', net.getValue('y'));
fprintf('Derivative of initial_x = %g\n', net.getDer('initial_x'));


% run network with 8 For-loop iterations
net.eval({'initial_x', 1, 'num_iterations', 8});

fprintf('Output y after 8 iterations = %g\n', net.getValue('y'));
fprintf('Derivative of initial_x = %g\n', net.getDer('initial_x'));

