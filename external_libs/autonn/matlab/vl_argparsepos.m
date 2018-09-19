function [opts, pos, unknown] = vl_argparsepos(opts, args, varargin)
%VL_ARGPARSEPOS Parse list of param.-value pairs, with positional arguments
%   [OPTS, POS] = VL_ARGPARSEPOS(OPTS, ARGS)
%   Same as VL_ARGPARSE, but allows arbitrary positional arguments before
%   the name-value pairs. The positional arguments are returned in the cell
%   array POS (see example below).
%
%   [OPTS, POS, UNKNOWN] = VL_ARGPARSEPOS(OPTS, ARGS)
%   Also returns unknown (non-positional) options in UNKNOWN. This is the
%   same as the second returned value from VL_ARGPARSE.
%
%   [OPTS, POS, UNKNOWN] = VL_ARGPARSEPOS(OPTS, ARGS, 'flags', FLAGS)
%   additionally scans the arguments list for single-string flags (as
%   opposed to the name-value pairs), and if found they are returned at the
%   end of the UNKNOWN list.
%
%   Any extra options (such as 'nonrecursive') are passed to VL_ARGPARSE as
%   well.
%
%   Example 1:
%     opts.pad = 0 ;
%     [opts, pos] = vl_argparsepos(opts, {x, y, 'pad', 1}) ;
%   Result:
%     opts.pad = 1
%     pos = {x, y}
%
%   Example 2:
%     opts.pad = 0 ;
%     [opts, pos, unknown] = vl_argparsepos(opts, {x, 'pad', 1, 'A', []}) ;
%   Result:
%     opts.pad = 1
%     pos = {x}
%     unknown = {'A', []}

% Copyright (C) 2016 Joao F. Henriques.
% All rights reserved.
%
% This file is part of the VLFeat library and is made available under
% the terms of the BSD license (see the COPYING file).

  % check for 'flags' option
  idx = find(strcmpi(varargin, 'flags'), 1) ;
  if ~isempty(idx)
    assert(idx + 1 <= numel(varargin) && iscellstr(varargin{idx + 1}), ...
      'Expected a cell array of strings after ''flags'' option.');
    
    flags = varargin{idx + 1} ;
    varargin(idx : idx + 1) = [] ;
    
    % scan args for each flag
    usedFlag = false(size(flags)) ;
    for i = 1:numel(flags)
      pos = find(strcmpi(flags{i}, args), 1) ;
      if ~isempty(pos)
        usedFlag(i) = true ;
        args(pos) = [] ;
      end
    end
  else  % no flags
    flags = {} ;
    usedFlag = [] ;
  end

  % even or odd indexes, always including the 2nd-to-last element of args
  idx = (numel(args) - 1 : -2 : 1) ;  % reverse order
  
  if isempty(idx)
    % no name-value pairs in the list
    firstPair = numel(args) + 1 ;
  else
    % find first invalid name-value pair, starting from the end
    pos = find(~cellfun(@ischar, args(idx)), 1) ;

    if isempty(pos)  % all are valid
      firstPair = idx(end) ;
    else  % map back to argument indexes
      firstPair = idx(pos) + 2 ;
    end
  end
  
  % separate them
  namedArgs = args(firstPair:end) ;
  pos = args(1:firstPair-1) ;

  % call vl_argparse
  if nargout >= 3
    [opts, unknown] = vl_argparse(opts, namedArgs, varargin{:}) ;
    unknown = [unknown, flags(usedFlag)] ;
  else
    opts = vl_argparse(opts, namedArgs, varargin{:}) ;
    assert(isempty(flags), 'Cannot specify flags with less than 3 return values.') ;
  end
  
end

