function results = run_MEEM(seq, res_path, bSaveImage)


res = MEEMTrack(seq.path,seq.ext,bSaveImage,seq.init_rect, seq.startFrame, seq.endFrame);

results.type   = 'rect';
results.res    = res.res;
results.fps    = res.fps;