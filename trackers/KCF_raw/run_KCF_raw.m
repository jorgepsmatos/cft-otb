function results = run_KCF(seq, res_path, bSaveImage)

base_path = './data/Benchmark/';

%default settings
kernel_type = 'gaussian';
feature_type = 'gray'; 
show_visualization = bSaveImage;


%parameters according to the paper. at this point we can override
%parameters based on the chosen kernel or feature type
kernel.type = kernel_type;

features.gray = false;
features.hog = false;

padding = 1.5;  %extra area surrounding the target
lambda = 1e-4;  %regularization
output_sigma_factor = 0.1;  %spatial bandwidth (proportional to target)

switch feature_type
case 'gray',
    interp_factor = 0.075;  %linear interpolation factor for adaptation

    kernel.sigma = 0.2;  %gaussian kernel bandwidth

    kernel.poly_a = 1;  %polynomial kernel additive term
    kernel.poly_b = 7;  %polynomial kernel exponent

    features.gray = true;
    cell_size = 1;

case 'hog',
    interp_factor = 0.02;

    kernel.sigma = 0.5;

    kernel.poly_a = 1;
    kernel.poly_b = 9;

    features.hog = true;
    features.hog_orientations = 9;
    cell_size = 4;

otherwise
    error('Unknown feature.')
end


assert(any(strcmp(kernel_type, {'linear', 'polynomial', 'gaussian'})), 'Unknown kernel.')


img_files = seq.s_frames;
% Seletected target size
target_sz = [seq.init_rect(1,4), seq.init_rect(1,3)];
% Initial target position
pos = [seq.init_rect(1,2), seq.init_rect(1,1)] + floor(target_sz/2); 
video_path = seq.path;
		
%call tracker function with all the relevant parameters
[positions, time] = tracker(video_path, img_files, pos, target_sz, ...
    padding, kernel, lambda, output_sigma_factor, interp_factor, ...
    cell_size, features, show_visualization);


targetSz = ones(seq.len,1)*target_sz;

results.type   = 'rect';
results.res    = [positions(:,2)-targetSz(:,2)/2 positions(:,1)-targetSz(:,1)/2 targetSz(:,2) targetSz(:,1)];
results.fps    = numel(img_files) / time;

