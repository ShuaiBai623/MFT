function [y, dzdx2] = vl_nncosineloss(x1, x2, c, varargin)
%VL_NNCOSINELOSS Compute cosine embedding loss
%   Y = VL_NNCOSINELOSS(X1, X2, C) computes the contrastive loss incurred by
%   the similar and dissimilar pairs in X1 and X2 labelled by C, making use
%   of the cosine similarity function to assess the distance between embeddings
%   (rather than euclidean metric used in the original contrastive loss
%   formulation of [1]).
%
%   The variables X1 nd X2 are of size H x W x D x N. The distance between
%   two pairs is computed between vectorised chunks of size HWD x N,
%   keeping the spatial and channel arrangement.
%
%   C has dimension 1 x 1 x 1 x N and specifies dissimilar pairs when
%   equal 0 and similar pairs otherwise.
%
%   The loss between two vectors X1 and X2 with cosine similarity
%   D = 1 - X1'X2 / (||X1||_2 *||X2||_2) and label C is computed as:
%   L(D, L) = sum(C * (1 - COS_SIM(X1,X2)) + (1-C) * max(COS_SIM(X1,X2) - M, 0)).
%
%   The term "cosine distance" is often used to denote the quantity
%   D = 1 - cos_sim(X1,X2), where cos_sim refers to the cosine similarity
%   between X1 and X2.  However, it is worth noting that this is not, strictly
%   speaking a distance function (it does not obey Cauchy-Schwarz). The
%   VL_NNCOSINELOSS seeks to minimise the cosine distance between similar pairs
%   and minimise cosine similarity between dissimilar pairs, up to a margin.
%
%   [DZDX1, DZDX2] = VL_NNCOSINELOSS(X1, X2, C, DZDY) computes the
%   derivative of the block projected onto the output derivative DZDY.
%   DZDX1, DZDX2 and DZDY have the same dimensions as X1, X2 and Y
%   respectively.
%
%   VL_NNCOSINELOSS(.., 'option', value, ...) accepts the following options:
%
%   `margin`:: 0.5
%    The maximum margin to be enforced between dissimilar pairs.
%
%    See also: VL_NNLOSS().
%
%    Based on the VL_NNCONTRLOSS() by Karel Lenc.
%
%    [1] Hadsell, Raia, Sumit Chopra, and Yann LeCun. "Dimensionality
%    reduction by learning an invariant mapping." CVPR 2006

% Copyright (C) 2018 Samuel Albanie
% Licensed under The MIT License [see LICENSE.md for details]

	opts.margin = 0.5 ;
  [opts, dzdy] = vl_argparsepos(opts, varargin) ;

	sx1 = size(x1) ; sx2 = size(x2) ; bsize = size(x1, 4) ;
	assert(numel(sx1) == numel(sx2), 'input sizes must match') ;
	assert(all(sx1 == sx2), 'Invalid input sizes.') ;
	assert(numel(c) == bsize, 'Invalid number of labels.') ;

  % allow element-specific margins
	if numel(opts.margin) > 1
		assert(numel(opts.margin) == bsize, 'Invalid margin.');
		opts.margin = reshape(opts.margin, [], bsize);
	end
	c = reshape(c, [], bsize);

  sims = vl_nncosinesim(x1, x2) ;
  diff = 1 - sims ;
	mdist = sims - opts.margin ;

	if isempty(dzdy)
		sims(c == 0) = max(mdist(c == 0), 0) ;
		y = sum(sims);
	else
    % outcome is invariant to order of projection
		[dcos1, dcos2] = vl_nncosinesim(x1, x2, dzdy{1}) ;
		keyboard
		one = ones(1, 'like', x1);
		mdist = squeeze(mdist);
		y1 = diff * (dzdy * 2);
		nf = mdist ./ (dist + 1e-4*one);
		neg_sel = mdist >  0 & c == 0;
		y1(:, neg_sel) = bsxfun(@times, -y1(:, neg_sel), nf(neg_sel));
		y1(:, mdist <= 0 & c == 0) = 0;
		y2 = -y1;
		y1 = reshape(y1, sx1);
		y2 = reshape(y2, sx2);
	end
