function [filename, epoch] = checkpoint(pattern)
%CHECKPOINT Returns the most recent model checkpoint, to resume training
%   FILENAME = models.checkpoint(PATTERN) returns the most recent model
%   checkpoint. FILENAME is the path to a MAT file that matches the string
%   PATTERN, which must have a single wildcard symbol '*'.
%
%   This is useful when the state of training (Net/Solver/Stats objects) is
%   periodically saved to MAT files so that it can be resumed in case of
%   failure, a technique called checkpointing. This function returns the
%   FILENAME of the last checkpoint, so that LOAD(FILENAME) restores the
%   objects.
%
%   For example, the pattern '/results/checkpoint*.mat' matches files
%   'checkpoint1.mat', 'checkpoint2.mat', etc, in a folder '/results'.
%   Calling this function will return the name of the last file, in
%   numerical order. Both zero-padded and unpadded numbers are supported.
%
%   See 'autonn/examples/cnn/cifar_example.m' for a complete example.
%
%   [FILENAME, EPOCH] = models.checkpoint(PATTERN) also returns EPOCH, the
%   number that matched the wildcard (e.g. 2 for 'checkpoint2.mat').

% Copyright (C) 2018 Joao F. Henriques.
% All rights reserved.
%
% This file is part of the VLFeat library and is made available under
% the terms of the BSD license (see the COPYING file).

  assert(nnz(pattern == '*') == 1, 'The file pattern must include exactly one wildcard (*).') ;
  
  % gather files respecting pattern
  files = dir(pattern) ;
  
  % replace wildcard with regexp to extract an integer from each file name
  [~, file_pattern] = fileparts(pattern);
  iterations = regexp({files.name}, strrep(file_pattern, '*', '([\d]+)'), 'tokens');
  
  % convert string integer to numeric, from doubly-nested cells
  iterations = cellfun(@(x) sscanf(x{1}{1}, '%d'), iterations);
  
  if isempty(iterations)
    % no files found
    filename = [] ;
    epoch = 1 ;
  else
    % return last file name, and starting epoch
    filename = strrep(pattern, '*', int2str(max(iterations))) ;
    epoch = max(iterations) + 1 ;
  end
end
