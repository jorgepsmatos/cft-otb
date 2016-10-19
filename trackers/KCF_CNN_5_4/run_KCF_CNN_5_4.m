function results = run_OURS_CNN(seq, res_path, bSaveImage)
%RUN_OURS Summary of this function goes here
%   Detailed explanation goes here
    reset(gpuDevice(1))

    setenv('LD_LIBRARY_PATH', '/usr/local/cuda/lib64:')
    setenv('CAFFE_ROOT', '/home/jorgematos/caffe');
    setenv('PYTHONPATH', '/home/jorgematos/caffe/python');

    imagesPathFile = fopen('imagesPath.txt','w');
    
    for iterator = 1:seq.len
        string = seq.s_frames{iterator};
        fprintf(imagesPathFile,'%s\n', string);
    end
    fclose(imagesPathFile);

    dlmwrite('initRect.txt', seq.init_rect)
    
    tic
    system('python main.py');
    duration = toc; 
    
    result = dlmread('output.txt');
    
    results.res = result;
    results.type='rect';
    results.fps=seq.len/duration;

    reset(gpuDevice(1))
end

