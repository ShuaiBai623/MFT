function results = testing_MFT(seq)

close all
addpath('../');
run setup_paths;
run vl_setupnn;
hog_params.cell_size = 4;
hog_params.compressed_dim = 31;

% grayscale_params.colorspace='gray';
% grayscale_params.cell_size = 1;

cn_params.tablename = 'CNnorm';
cn_params.useForGray = false;
cn_params.cell_size = 4;
cn_params.compressed_dim = 10;

ic_params.tablename = 'intensityChannelNorm6';
ic_params.useForColor = false;
ic_params.cell_size = 4;
ic_params.compressed_dim = 3;

cnn_params.nn_name = 'imagenet-resnet-50-dag.mat'; % Name of the network
cnn_params.output_layer = { 'conv1x' 'res3d' 'res4f'};

cnn_params.downsample_factor = [2  1 1];           % How much to downsample each output layer
cnn_params.compressed_dim = [64 256 256];            % Compressed dimensionality of each output layer
cnn_params.input_size_mode = 'adaptive';        % How to choose the sample size
cnn_params.input_size_scale = 1;                % Extra scale factor of the input samples to the network (1 is no scaling)

% Which features to include

s_frames = seq.s_frames;
params.ground_truth = seq.ground_truth
% cnn_params.nn_name = 'imagenet-matconvnet-vgg-verydeep-16.mat'; % Name of the network
% cnn_params.output_layer = [2 14 20];
% cnn_params.downsample_factor = [2 1 1];           % How much to downsample each output layer
% cnn_params.compressed_dim = [64 256 128];          % Compressed dimensionality of each output layer
% cnn_params.input_size_mode = 'adaptive';        % How to choose the sample size
% cnn_params.input_size_scale = 1;                % Extra scale factor of the input samples to the network (1 is no scaling)
       % Extra scale factor of the input samples to the network (1 is no scaling)
cnn_params.use_gpu = true;
% cnn_params.return_gpu = true;
params.use_gpu = true; 
cnn_params.gpu_id = [1];	

cnn_params3.nn_name = 'SE-ResNet-50-mcn.mat'; % Name of the network
cnn_params3.output_layer = {'conv1_7x7_s2xxx' 'conv3_4x' 'conv4_6x'};


cnn_params3.downsample_factor = [2 1 1];           % How much to downsample each output layer
cnn_params3.compressed_dim = [64 256 256];          % Compressed dimensionality of each output layer
cnn_params3.input_size_mode = 'adaptive';        % How to choose the sample size
cnn_params3.input_size_scale = 1;                % Extra scale factor of the input samples to the network (1 is no scaling)
cnn_params3.use_gpu = true;
% cnn_params.return_gpu = true;
cnn_params3.gpu_id = [1];
% Which features to include
params.t_features = {
    ...struct('getFeature',@get_cnn_layers_drag, 'fparams',cnn_params),...
    struct('getFeature',@get_cnn_layers_drag, 'fparams',cnn_params3),...
    ...struct('getFeature',@get_colorspace, 'fparams',grayscale_params),...
    ...struct('getFeature',@get_fhog,'fparams',hog_params),...
    ...struct('getFeature',@get_table_feature, 'fparams',cn_params),...
    ...struct('getFeature',@get_table_feature, 'fparams',ic_params),...
};

% Global feature parameters1s
params.t_global.normalize_power = 2;    % Lp normalization with this p
params.t_global.normalize_size = true;  % Also normalize with respect to the spatial size of the feature
params.t_global.normalize_dim = true;   % Also normalize with respect to the dimensionality of the feature

% Image sample parameters
params.search_area_shape = 'square';    % The shape of the samples
params.search_area_scale = 4;         	% The scaling of the target size to get the search area
params.min_image_sample_size = 224^2;   % Minimum area of image samples
params.max_image_sample_size = 250^2;   % Maximum area of image samples

% Detection parameters
params.refinement_iterations = 1;       % Number of iterations used to refine the resulting position in a frame
params.newton_iterations = 5;           % The number of Newton iterations used for optimizing the detection score
params.clamp_position = false;          % Clamp the target position to be inside the image

% Learning parameters
params.output_sigma_factor = { 1/12 1/12 1/4 };		% Label function sigma
params.index_all={[1 2] [3] };	% Label function sigma
params.learning_rate = 0.012;	 	 			% Learning rate
params.nSamples = 100;          			  % Maximum number of stored training samples
params.sample_replace_strategy = 'lowest_prior';    % Which sample to replace when the memory is full
params.lt_size = 0;                     % The size of the long-term memory (where all samples have equal weight)
params.train_gap = 5;                   % The number of intermediate frames with no training (0 corresponds to training every frame)
params.skip_after_frame = 1;            % After which frame number the sparse update scheme should start (1 is directly)
params.use_detection_sample = true;     % Use the sample that was extracted at the detection stage also for learning

% Factorized convolution parameters
params.use_projection_matrix = true;    % Use projection matrix, i.e. use the factorized convolution formulation
params.update_projection_matrix = true; % Whether the projection matrix should be optimized or not
params.proj_init_method = 'pca';        % Method for initializing the projection matrix
params.projection_reg = 2e-7;	 	 	% Regularization paremeter of the projection matrix

% Generative sample space model parameters
params.use_sample_merge = true;                 % Use the generative sample space model to merge samples
params.sample_update_criteria = 'Merge';        % Strategy for updating the samples
params.weight_update_criteria = 'WeightedAdd';  % Strategy for updating the distance matrix
params.neglect_higher_frequency = false;        % Neglect hiigher frequency components in the distance comparison for speed

% Conjugate Gradient parameters
params.CG_iter = 5;                     % The number of Conjugate Gradient iterations in each update after the first frame
params.init_CG_iter = 10*20;            % The total number of Conjugate Gradient iterations used in the first frame
params.init_GN_iter = 10;               % The number of Gauss-Newton iterations used in the first frame (only if the projection matrix is updated)
params.CG_use_FR = false;               % Use the Fletcher-Reeves (true) or Polak-Ribiere (false) formula in the Conjugate Gradient
params.CG_standard_alpha = true;        % Use the standard formula for computing the step length in Conjugate Gradient
params.CG_forgetting_rate = 75;	 	 	% Forgetting rate of the last conjugate direction
params.precond_data_param = 0.7;	 	% Weight of the data term in the preconditioner
params.precond_reg_param = 0.1;	 	% Weight of the regularization term in the preconditioner
params.precond_proj_param = 30;	 	 	% Weight of the projection matrix part in the preconditioner

% Regularization window parameters
params.use_reg_window = true;           % Use spatial regularization or not
params.reg_window_min = 1e-4;			% The minimum value of the regularization window
params.reg_window_edge = 10e-3;         % The impact of the spatial regularization
params.reg_window_power = 2;            % The degree of the polynomial to use (e.g. 2 is a quadratic window)
params.reg_sparsity_threshold = 0.12;   % A relative threshold of which DFT coefficients that should be set to zero

% Interpolation parameters
params.interpolation_method = 'bicubic';    % The kind of interpolation kernel
params.interpolation_bicubic_a = -0.75;     % The parameter for the bicubic interpolation kernel
params.interpolation_centering = true;      % Center the kernel at the feature sample
params.interpolation_windowing = false;     % Do additional windowing on the Fourier coefficients of the kernel

% Scale parameters for the translation model
% Only used if: params.use_scale_filter = false
params.use_scale_filter = false
params.number_of_scales = 10;            % Number of scales to run the detector
params.scale_step = 1.03;               % The scale factor

% Initialize
params.init_sz = [seq.init_rect(1,4), seq.init_rect(1,3)];
params.init_pos = [seq.init_rect(1,2), seq.init_rect(1,1)] + (params.init_sz - 1)/2;
params.s_frames = s_frames;

% Run tracker
results = tracker(params);
