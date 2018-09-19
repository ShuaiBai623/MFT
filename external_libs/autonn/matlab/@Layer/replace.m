function replace(obj, originals, replacements)
%REPLACE Performs in-place replacement of layers with new layers
%   OBJ.REPLACE(ORIGINAL, NEW) replaces each reference to a layer ORIGINAL
%   with a different layer NEW. The replacement is done in all Layer
%   objects involved in the computation of OBJ, by modifying their input
%   arguments.
%
%   OBJ.REPLACE({ORIG1, ORIG2, ...}, {NEW1, NEW2, ...}) replaces layer
%   ORIG1 with NEW1, ORIG2 with NEW2, etc.
%
%   Note that, while the original values must be Layer objects, the new
%   values can be anything (including constant values).

% Copyright (C) 2017 Joao F. Henriques.
% All rights reserved.
%
% This file is part of the VLFeat library and is made available under
% the terms of the BSD license (see the COPYING file).

  if ~iscell(originals)
    originals = {originals} ;
  end
  if ~iscell(replacements)
    replacements = {replacements} ;
  end

  assert(numel(originals) == numel(replacements), 'Input cell arrays must have the same length.') ;
  
  % gather all layers
  layers = find(obj) ;

  for i = 1:numel(originals)
    o = originals{i} ;
    r = replacements{i} ;
    
    % do the replacement in the layers' inputs lists
    for j = 1:numel(layers)
      for k = 1:numel(layers{j}.inputs)
        if eq(o, layers{j}.inputs{k}, 'sameInstance')
          layers{j}.inputs{k} = r ;
        end
      end
    end
  end

end

