function results = run_MDNET(seq, res_path, bSaveImage)

if(isempty(gcp('nocreate')))
    parpool;
end

run matconvnet/matlab/vl_setupnn ;

addpath('pretraining');
addpath('tracking');
addpath('utils');

%config.imgList = seq.s_frames; 
%config.gt = dlmread(['../../anno/' regexprep(seq.name,'_.*','') '.txt']);

net = fullfile('models','mdnet_vot-otb.mat');

tic
result = mdnet_run(seq.s_frames, seq.init_rect, net, true);
duration = toc;

results.res = result;
results.type='rect';
results.fps=seq.len/duration;

