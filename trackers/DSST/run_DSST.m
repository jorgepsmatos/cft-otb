function results = run_DSST(seq, res_path, bSaveImage)

%parameters according to the paper
params.padding = 1.0;         			% extra area surrounding the target
params.output_sigma_factor = 1/16;		% standard deviation for the desired translation filter output
params.scale_sigma_factor = 1/4;        % standard deviation for the desired scale filter output
params.lambda = 1e-2;					% regularization weight (denoted "lambda" in the paper)
params.learning_rate = 0.025;			% tracking model learning rate (denoted "eta" in the paper)
params.number_of_scales = 33;           % number of scale levels (denoted "S" in the paper)
params.scale_step = 1.02;               % Scale increment factor (denoted "a" in the paper)
params.scale_model_max_area = 512;      % the maximum size of scale examples

params.visualization = bSaveImage;

img_files = seq.s_frames;
% Seletected target size
target_sz = [seq.init_rect(1,4), seq.init_rect(1,3)];
% Initial target position
pos       = [seq.init_rect(1,2), seq.init_rect(1,1)]; % + floor(target_sz/2); 

params.init_pos = floor(pos); %+ floor(target_sz/2);
params.wsize = floor(target_sz);
params.img_files = img_files;
params.video_path = seq.path;

[positions, fps] = dsst(params);

%rects      =   [positions(:,2) - target_sz(2)/2, positions(:,1) - target_sz(1)/2, positions(:,4), positions(:,3)];

results.type   = 'rect';
results.res    = [positions(:,2) positions(:,1) positions(:,4) positions(:,3)];
results.fps    = fps;