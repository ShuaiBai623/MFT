% init

rng(0) ;
C = 3 ;
x = randn(5,5,C,2) ;
g = rand(1,1,C) ;
b = rand(1,1,C) ;
clip = [1 0] ;

moments = rand(C, 2) ;

%ym1 = vl_nnbnorm(x, g, b, 'moments', moments) ;
test = false ;
ym1 = vl_nnbnorm(x, g, b) ;
ym2 = vl_nnbrenorm(x, g, b, moments, clip, test) ;

r1 = norm(ym1(:)) ;
r2 = norm(ym2(:)) ;

mdiff = ym2 - ym1 ;

fprintf('moments diff norm: %f\n', norm(mdiff(:))) ;

% in test mode, moments are used
moments = rand(C, 2) ;

%ym1 = vl_nnbnorm(x, g, b, 'moments', moments) ;
test = true ;
ym1 = vl_nnbnorm(x, g, b, 'moments', moments) ;
ym2 = vl_nnbrenorm(x, g, b, moments, clip, test) ;

r1 = norm(ym1(:)) ;
r2 = norm(ym2(:)) ;

mdiff = ym2 - ym1 ;

fprintf('moments test mode diff norm: %f\n', norm(mdiff(:))) ;

% -------------------------------------------------------------------
%                                         Backwards mode with moments
% -------------------------------------------------------------------

% check ders
dzdy = rand(size(x)) ;

test = false ;
[dzdx1, dzdg1, dzdb1] = vl_nnbnorm(x, g, b, dzdy) ;
%[dzdx1, dzdg1, dzdb1] = vl_nnbnorm(x, g, b, dzdy, 'moments', moments) ;
[dzdx2, dzdg2, dzdb2] = vl_nnbrenorm(x, g, b, moments, clip, test, dzdy) ;

ddiff = dzdx2 - dzdx1 ;
fprintf('diff der x: %f\n', norm(ddiff(:))) ;

ddiffg = squeeze(dzdg2) - dzdg1 ;
fprintf('diff der g: %f\n', norm(ddiffg(:))) ;

ddiffb = squeeze(dzdb2) - dzdb1 ;
fprintf('diff der b: %f\n', norm(ddiffb(:))) ;
