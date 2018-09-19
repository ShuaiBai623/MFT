function workspaceNames(modifier)
%WORKSPACENAMES Sets names of unnamed layers based on the current workspace
%   Layer.workspaceNames() replaces empty layer names with new names, based
%   on the names of the corresponding variables in the caller's workspace.
%
%   Layer.workspaceNames(MODIFIER) also specifies a function handle to be
%   evaluated on each name, possibly modifying it (e.g. to append a prefix
%   or suffix).
%
%   See also Layer.sequentialNames.
%
%   Example:
%     images = Input() ;
%     Layer.workspaceNames() ;
%     images.name  % returns 'images'

  if nargin < 1, modifier = @deal ; end

  varNames = evalin('caller','who') ;
  for i = 1:numel(varNames)
    layer = evalin('caller', varNames{i}) ;
    if isa(layer, 'Layer') && isempty(layer.name)
      layer.name = modifier(varNames{i}) ;
    end
  end
end

