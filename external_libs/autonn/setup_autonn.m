function setup_autonn(flag)
%SETUP_AUTONN Sets up AutoNN
%   SETUP_AUTONN sets up AutoNN, by adding its folders to the Matlab path.
%
%   Note that MatConvNet should also be on the path, by calling VL_SETUPNN.
%
%   SETUP_AUTONN('silent') does not print any messages.

  firstTime = ~exist('Layer', 'file') ;

  % add things to the path
  root = fileparts(mfilename('fullpath')) ;
  addpath(root, [root '/matlab'], [root '/matlab/wrappers'], [root '/matlab/derivatives']) ;
  
  if nargin == 0 || ~strcmp(flag, 'silent')
    % first time message with documentation link
    if firstTime
      [~, folder] = fileparts(root) ;  % folder should be just 'autonn', but no guarantees
      if ~usejava('desktop')
        disp(['AutoNN is set up. (Documentation: help ' folder ')']) ;
      else
        disp(['AutoNN is set up. (Documentation: <a href="matlab:help ' ...
          folder '">help ' folder '</a>)']) ;
      end
    end

    % warn if MatConvNet is missing
    if ~exist('vl_setupnn', 'file')
      warning('AutoNN:MatConvNetMissing', ['MatConvNet is not on the path. '...
        'You can set it up by calling:\n  run MATCONVNET/vl_setupnn\n' ...
        'replacing ''MATCONVNET'' with the directory where it is located.']) ;
    end
  end
end

