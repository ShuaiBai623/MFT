function displayCustom(obj, varName, showLinks)
% DISPLAYCUSTOM(OBJ)
% Customizes the display of the Layer's contents, shown here as a function
% call (e.g. 'func(a, b)'). Also shows hyperlinks in the command window,
% allowing one to interactively traverse the network.
%
% This method is meant to be overriden by subclasses, like Input or Param.

% Copyright (C) 2016 Joao F. Henriques.
% All rights reserved.
%
% This file is part of the VLFeat library and is made available under
% the terms of the BSD license (see the COPYING file).

  if nargin < 2
    varName = [] ;
  end
  if nargin < 3
    showLinks = false ;
  end

  fprintf('%s(', char(obj.func)) ;

  for i = 1:numel(obj.inputs)
    input = obj.inputs{i} ;

    if ~isa(input, 'Layer')
      if isnumeric(input) && isscalar(input)
        % a scalar, display it
        fprintf('%g', input) ;
      elseif isnumeric(input) && isvector(input) && numel(input) <= 8 && ~isempty(input)
        % a short numeric vector, display it inline
        fprintf('[%g', input(1)) ;
        fprintf(' %g', input(2:end)) ;
        fprintf(']') ;
        if size(input,1) > 1, fprintf('''') ; end  % column vector
      else
        % use Matlab's native display of single cells, which provides a
        % nice short representation of any object (e.g. '[3x3 double]')
        fprintf(strtrim(evalc('disp({input})'))) ;
      end
    else
      % another layer, display it along with a navigation hyperlink
      if ~isempty(input.name)
        label = input.name ;
      elseif isa(input, 'Input')
        label = 'Input' ;
      elseif isa(input, 'Param')
        label = sprintf('Param(%s)', strtrim(evalc('disp({input.value})'))) ;
      else
        label = sprintf('inputs{%i}', i) ;
      end

      if ~showLinks || isempty(varName)
        fprintf(label) ;
      else
        cmd = sprintf('%s.inputs{%i}', varName, i) ;
        fprintf('<a href="matlab:display(%s,''%s'')">%s</a>', cmd, cmd, label) ;
      end
    end
    if i < numel(obj.inputs)
      fprintf(', ') ;
    end
  end
  fprintf(') ') ;
end

