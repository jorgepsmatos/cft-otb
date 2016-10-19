
% This demo script runs the SRDCF tracker on the included "Couple" video.

% Load video information
video_path = 'sequences/Couple';
[seq, ~] = load_video_info(video_path);

% Run SRDCF
results = run_SRDCF(seq);