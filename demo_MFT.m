
% This demo script runs the MFT tracker with deep features on the
% included "Crossing" video.

% Add paths
setup_paths();
warning('off') 
% Load video information
video_path = 'sequences/Crossing/';
[seq, ground_truth] = load_video_info(video_path);
all_iou=0;
% Run MFT
seq.ground_truth=ground_truth
results = testing_MFT(seq);
disp(size(results.res))
disp(size(ground_truth))

% Run MFT
for i =1:size(ground_truth,1)

    all_iou = all_iou +overlap_ratio(results.res(i,1:4),ground_truth(i,1:4));
end
disp(['all_ave ' num2str(all_iou/i) ]);

