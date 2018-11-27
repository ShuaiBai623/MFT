function test_mcnExtraLayers(varargin)
% run tests for ExtraLayers module

  opts.dev = false ;
  opts = vl_argparse(opts, varargin) ;

  % add tests to path
  addpath(fullfile(fileparts(mfilename('fullpath')), 'matlab/xtest')) ;
  addpath(fullfile(vl_rootnn, 'matlab/xtest/suite')) ;

  % test network layers
  run_extra_layers_tests('command', 'nn', 'dev', opts.dev) ;
end
