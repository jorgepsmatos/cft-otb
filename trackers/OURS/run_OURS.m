function results = run_OURS(seq, res_path, bSaveImage)
%RUN_OURS Summary of this function goes here
%   Detailed explanation goes here
    imagesPathFile = fopen('imagesPath.txt','w');
    
    for iterator = 1:seq.len
        string = seq.s_frames{iterator};
        fprintf(imagesPathFile,'%s\n', string);
    end
    fclose(imagesPathFile);

    dlmwrite('initRect.txt', seq.init_rect)
    
    tic
    system('python main.py -f HoG');
    duration = toc; 
    
    result = dlmread('output.txt');
    
    results.res = result;
    results.type='rect';
    results.fps=seq.len/duration;
end

