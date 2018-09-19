
# AutoNN #
AutoNN is a functional wrapper for [MatConvNet](http://www.vlfeat.org/matconvnet/), implementing automatic differentiation.

It builds on MatConvNet's low-level functions and Matlab's math operators, to create a modern deep learning API with automatic differentiation at its core. The guiding principles are:

- Concise syntax for fast research prototyping, mixing math and deep network blocks freely.
- No boilerplate code to create custom layers, implemented as Matlab functions operating on GPU arrays.
- Minimal execution kernel for backpropagation, with a focus on speed.

Compared to the SimpleNN and DagNN [wrappers](http://www.vlfeat.org/matconvnet/wrappers/) for MatConvNet, AutoNN is less verbose and has lower computational overhead.


# Requirements #

* A recent Matlab (preferably 2016b onwards, though older versions may also work).
* MatConvNet (preferably the [most recent version](https://github.com/vlfeat/matconvnet), though others may still work).


# AutoNN in a nutshell #

Defining an objective function with AutoNN is as simple as:

```Matlab
% define inputs and learnable parameters
x = Input() ;
y = Input() ;
w = Param('value', randn(1, 100)) ;
b = Param('value', 0) ;

% combine them using math operators, which define the prediction
prediction = w * x + b ;

% define a loss
loss = sum(sum((prediction - y).^2)) ;

% compile and run the network
net = Net(loss) ;
net.eval({x, rand(100, 1), y, 0.5}) ;

% display parameter derivatives
net.getDer(w)
```

AutoNN also allows you to use MatConvNet layers and custom functions.

Here's a simplified 20-layer ResNet:

```Matlab
images = Input() ;

% initial convolution
x = vl_nnconv(images, 'size', [3 3 3 64], 'stride', 4) ;

% iterate blocks
for k = 1:20
  % compose a residual block, based on the previous output
  res = vl_nnconv(x, 'size', [3 3 64 64], 'pad', 1) ;  % convolution
  res = vl_nnbnorm(res) ;  % batch-normalization
  res = vl_nnrelu(res) ;  % ReLU
  
  % add it to the previous output
  x = x + res ;
end

% pool features across spatial dimensions, and do final prediction
pooled = mean(mean(x, 1), 2) ;
prediction = vl_nnconv(pooled, 'size', [1 1 64 1000]) ;
```

All of MatConvNet's layer functions are overloaded, as well as a growing list of Matlab math operators and functions. The derivatives for these functions are defined whenever possible, so that they can be composed to create differentiable models. A full list can be found [here](doc/methods.txt).

Finally, there are several classes to aid training, such as standard datasets, solvers, models, and statistics plotting. It is easy to mix them, and you retain full control over the training loop. For example:

```Matlab
% load dataset
dataset = datasets.CIFAR10('/data/cifar') ;

% create solver
solver = solvers.Adam() ;
solver.learningRate = 0.0001 ;

for epoch = 1:100  % iterate epochs
  for batch = dataset.train()  % iterate batches
    % draw samples
    [images, labels] = dataset.get(batch) ;

    % evaluate network to compute gradients
    net.eval({'images', images, 'labels', labels}) ;

    % take one gradient descent step
    solver.step(net) ;
  end
end
```

# Documentation #

## Tutorial ##

The easiest way to learn more is to follow this short [tutorial](doc/TUTORIAL.md). It covers all the basic concepts and a good portion of the API.


## Help pages ##

Comprehensive documentation is available by typing `help autonn` into the Matlab console. This lists all the classes and methods, with short descriptions, and provides links to other help pages.


## Converting SimpleNN/DagNN models ##

For a quicker start or to load pre-trained models, you may want to import them from the existing wrappers. Check `help Layer.fromDagNN`.


## Examples ##

The `examples` directory has heavily-commented samples. These can be grouped in two categories:

- The *minimal* examples (in `examples/minimal`) are very short and self-contained. They are scripts so you can inspect and explore the resulting variables in the command window.

- The *full* examples (in `examples/cnn` and `examples/rnn`) demonstrate usage of the AutoNN training packages. These include several standard solvers (e.g. Adam, AdaGrad), CNN models (including automatically downloading pre-trained models), and datasets (e.g. ImageNet, CIFAR-10). You can override the parameters on the command window, and experiment with different models and settings.


# Screenshots #

Some gratuitous screenshots, though the important bits are in the code above really:

*Training diagnostics plot*

![Diagnostics](doc/diagnostics.png)

*Graph topology plot*

![Graph](doc/graph.png)


# Authors #

[AutoNN](https://en.wikipedia.org/wiki/Auton) was developed by [João F. Henriques](http://www.robots.ox.ac.uk/~joao/) at the [Visual Geometry Group (VGG)](http://www.robots.ox.ac.uk/~vgg/), University of Oxford.

We gratefully acknowledge contributions by: [Sam Albanie](http://www.robots.ox.ac.uk/~albanie/), Ryan Webster, [Ankush Gupta](http://www.robots.ox.ac.uk/~ankush/), [David Novotny](http://www.robots.ox.ac.uk/~david/), [Aravindh Mahendran](http://users.ox.ac.uk/~newc4521/index.html), Stefano Woerner.

