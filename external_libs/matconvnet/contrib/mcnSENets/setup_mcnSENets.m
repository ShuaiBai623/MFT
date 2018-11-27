function setup_mcnSENets()
%SETUP_MCNSENETS Sets up mcnSENets, by adding its folders 
% to the Matlab path
%
% Copyright (C) 2017 Samuel Albanie
% Licensed under The MIT License [see LICENSE.md for details]

  root = fileparts(mfilename('fullpath')) ;
  addpath(root, [root '/matlab'], [root '/benchmarks'], [root '/misc']) ;
  addpath([vl_rootnn '/examples/imagenet'], [vl_rootnn, '/examples']) ;
