function results=run_ASMS(seq, res_path, bSaveImage)

% Create region.txt with initial bounding box
regionFileID = fopen('region.txt','w');
fprintf(regionFileID, '%d,%d,%d,%d\n', seq.init_rect(1), seq.init_rect(2), seq.init_rect(3), seq.init_rect(4));
fclose(regionFileID);

% Get the path to all video files
videoSequenceFolderInfo = dir([seq.path '*.jpg']);

% Cell array with complete system path to each image
for ii = 1:seq.len
    pathToImages{ii,1} = strcat(seq.path, videoSequenceFolderInfo(ii).name);
end

% Write the images path to file images.txt
imagesFileID = fopen('images.txt','w');
for row = 1:seq.len
    fprintf(imagesFileID, '%s\n', pathToImages{row,:});
end 
fclose(imagesFileID);

tic
[status, result] = system('./asms_vot');
duration = toc;

results.res = dlmread('output.txt');
%results.res(:,1:2) =results.res(:,1:2) + 1;%c to matlab

results.type='rect';
results.fps=seq.len/duration;