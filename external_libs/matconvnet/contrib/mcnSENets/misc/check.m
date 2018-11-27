% scratch script - sanity checks for imported models

%net = load('data/models-import/SE-ResNeXt-50-mcn.mat') ;
%net = load('data/models-import/SE-ResNet-50-mcn.mat') ;
%net = load('data/models-import/SE-ResNet-101-mcn.mat') ;
%net = load('data/models-import/imagenet-resnet-50-dag.mat') ;
net = load('data/models-import/SE-BN-Inception-mcn.mat') ;
dag = dagnn.DagNN.loadobj(net) ;

labelMapPath = fullfile(vl_rootnn, 'contrib/mcnSENets/misc/label_map.txt') ;
labelMap = importdata(labelMapPath) ; 

imPath = 'peppers.png' ;
im = single(imresize(imread(imPath), [224 224])) ;
RGB = [123, 117, 104] ;
rgb = permute(RGB, [1 3 2]) ;
im = bsxfun(@minus, im, rgb) ;
dag.mode = 'test' ;
dag.eval({'data', im}) ;
preds = dag.vars(dag.getVarIndex('prob')).value ; 

[bestScore, best_] = max(preds) ;
best = find(labelMap == best_) ; % remap label
figure(1) ; clf ; imagesc(im) ; zs_dispFig ;
fprintf('%s (%d), score %.3f\n',...
dag.meta.classes.description{best}, best, bestScore) ;
