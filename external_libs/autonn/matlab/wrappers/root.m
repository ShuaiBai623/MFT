function out = root(varargin)
%ROOT Root layer for networks with multiple inputs
%   This is the last (root) layer of the network, that binds together
%   multiple network outputs if they exist. It does not actually output
%   their values because this is not needed.
%
%   The root layer exists purely to simplify some internal logic; there
%   should be no need to create root layers outside of Net.compile.

  out = [] ;

end

