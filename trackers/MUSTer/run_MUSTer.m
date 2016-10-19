function results = run_MEEM(seq, res_path, bSaveImage)
%RUN_MUSTER Summary of this function goes here
%   Detailed explanation goes here


source.video_path = seq.path;
source.n_frames = seq.len;
rect_init = seq.init_rect;
for ii=1:seq.len
    source.img_files{1,ii} = seq.s_frames{ii}(length(seq.path)+1:end);
end

tic
try
    bboxes = MUSTer_tracking(source, rect_init);
catch err
    bboxes = ones(seq.len, 4);
end    
duration = toc;

results.type   = 'rect';
results.res    = bboxes;
results.fps    = seq.len / duration;

end

