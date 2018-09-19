% AutoNN - Table of Contents
%
% Examples
%   examples/minimal   - Directory with minimal examples: regression and LSTM (start here)
%   examples/cnn       - Directory with CNN examples: ImageNet, CIFAR-10, MNIST and custom data
%   examples/rnn       - Directory with RNN/LSTM language model example on Shakespeare text
%
% Base classes
%   Layer              - Main building block for defining new networks
%   Net                - Compiled network that can be evaluated on data
%   Input              - Defines a network input (such as images or labels)
%   Param              - Defines a learnable network parameter
%   Selector           - Selects a single output of a multiple-outputs layer
%   Var                - Defines a network variable explicitly
%
% Training classes and packages
%   models             - Standard models package (e.g. AlexNet, VGG)
%   solvers            - Solvers package (e.g. SGD, Adam)
%   datasets           - Standard/custom datasets package (e.g. CIFAR-10)
%   Stats              - Aggregation and plotting of training statistics
%
% Extra CNN blocks (in addition to MatConvNet's)
%   vl_nnlstm          - Long Short-Term Memory cell (LSTM)
%   vl_nnlstm_params   - Initialize the learnable parameters for an LSTM
%   vl_nnaffinegrid    - Affine grid generator for Spatial Transformer Networks
%   vl_nnmaxout        - CNN maxout operator
%   vl_nnmask          - CNN dropout mask generator
%   For                - Differentiable For-loop or recursion, with dynamic iteration count
%   While              - Differentiable While-loop or recursion, with dynamic stop condition
%
% Layer methods
%   display            - Display layer information
%   find               - Searches for layers that meet the specified criteria
%   deepCopy           - Copies a network or subnetwork, optionally sharing some layers
%   evalOutputSize     - Computes output size of a layer
%   plotPDF            - Displays the network topology in a PDF
%   workspaceNames     - Sets names of unnamed layers based on the current workspace
%   sequentialNames    - Sets names of all unnamed layers based on type and order
%
% Layer overloaded methods
%   Operators          - Many functions and operators are overloaded, see: methods('Layer')
%   MatConvNet layers  - All MatConvNet layer functions are overloaded, see: methods('Layer')
%   vl_nnconv          - Additional options for vl_nnconv (CNN convolution)
%   vl_nnconvt         - Additional options for vl_nnconvt (CNN deconvolution)
%   vl_nnbnorm         - Additional options for vl_nnbnorm (CNN batch normalisation)
%   vl_nndropout       - Additional options for vl_nndropout (CNN dropout)
%   eq                 - Overloaded equality operator, or test for Layer instance equality
%
% Layer static methods
%   fromDagNN          - Converts a DagNN object to the AutoNN framework
%   fromCompiledNet    - Decompiles a Net back into Layer objects
%   fromFunction       - Generator for new custom layer type
%   create             - Creates a layer from a function handle and arguments
%
% Net methods
%   eval               - Network evaluation, including backpropagation to compute derivatives
%   displayVars        - Display table with information on variables and derivatives
%   getVarsInfo        - Retrieves network variables information as a struct
%   plotDiagnostics    - Creates or updates diagnostics plot
%   setParameterServer - Sets up a parameter server for multi-GPU training
%
% Utilities
%   setup_autonn       - Sets up AutoNN, by adding its folders to the Matlab path
%   cnn_benchmark      - Times execution of AutoNN and DagNN models
%   vl_argparsepos     - Parse list of param.-value pairs, with positional arguments
%   vl_parseprop       - Parses name-value pairs list to override properties of object
%   dynamic_subplot    - Dynamically reflowing subplots, to maintain aspect ratio
