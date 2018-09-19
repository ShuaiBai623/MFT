function unknown_args = vl_parseprop(obj, args, properties)
%VL_PARSEPROP Parses name-value pairs list to override properties of object
%   VL_PARSEPROP(OBJ, ARGS, PROPERTIES) is similar to VL_ARGPARSE, except
%   that it is meant to be used as a helper for class constructors.
%
%   An object OBJ has properties with default values. VL_PARSEPROP parses a
%   list of name-value pairs given as cell array ARGS, which are meant to
%   override them (e.g. specified by the user). Each pair (NAME, VALUE)
%   specifies that the assignment OBJ.NAME = VALUE will be made. The list
%   of valid properties is given in PROPERTIES.

% Copyright (C) 2018 Joao F. Henriques.
% All rights reserved.
%
% This file is part of the VLFeat library and is made available under
% the terms of the BSD license (see the COPYING file).

  assert(iscellstr(properties)) ;
  
  assert(mod(numel(args), 2) == 0, 'Expected name-value pairs.') ;
  
  names = args(1:2:end-1) ;
  values = args(2:2:end) ;
  assert(iscellstr(names), 'Expected name-value pairs.') ;
  
  unmatched = true(1, numel(names)) ;

  for i = 1:numel(properties)
    pos = find(strcmpi(names, properties{i})) ;
    
    if ~isempty(pos)  % use last value if more than one is given (i.e., override)
      obj.(properties{i}) = values{pos(end)} ;
    end
    unmatched(pos) = false ;
  end
  
  if nargout == 0
    if any(unmatched)
      error('Unknown argument: %s', names{find(unmatched, 1)}) ;
    end
  else
    unknown_args = [names(unmatched) ; values(unmatched)] ;
    unknown_args = unknown_args(:)' ;
  end
end

