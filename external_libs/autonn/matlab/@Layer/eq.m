function y = eq(a, b, same)
%EQ Overloaded equality operator, or test for Layer instance equality
%   EQ(A, B), A == B returns a Layer that tests equality of the outputs of
%   two Layers (one of them may be constant).
%
%   EQ(A, B, 'sameInstance') checks if two variables refer to the same
%   Layer instance (i.e., calls the == operator for handle classes).

  if nargin <= 2
    y = Layer(@eq, a, b) ;
    y.numInputDer = 0 ;  % non-differentiable
  else
    assert(isequal(same, 'sameInstance'), ...
      'The only accepted extra flag for EQ is ''sameInstance''.') ;
    
    if ~isa(a, 'Layer') || ~isa(b, 'Layer')
      y = false ;
    else
      y = eq@handle(a, b) ;
    end
  end
end

