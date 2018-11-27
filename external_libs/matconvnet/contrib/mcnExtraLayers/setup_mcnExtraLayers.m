function setup_mcnExtraLayers
%SETUP_MCNEXTRALAYERS Sets up mcnExtraLayers by adding its folders to the path

  root = fileparts(mfilename('fullpath')) ;
  addpath(root, [root '/matlab'], [root '/matlab/wrappers'], [root '/utils']) ;
