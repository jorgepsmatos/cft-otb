clear
addpath('./util');

seqs=configSeqs;

trks=configTrackers;



% TRE FPS
pathRes = './results/results_TRE_CVPR13/';

for index_algrm=1:length(trks)
    trks{index_algrm}.fpsTRE = [];
end

for index_seq=1:length(seqs)
    seq = seqs{index_seq};
    seq_name = seq.name;
    
    for index_algrm=1:length(trks)
        algrm = trks{index_algrm};
        name=algrm.name;
        
        fileName = [pathRes seq_name '_' name '.mat'];
        load(fileName); 
        
        for index_results = 1:length(results)
            if length(trks{index_algrm}.fpsTRE) == 0 
                trks{index_algrm}.fpsTRE = results{index_results}.fps;
            else
                trks{index_algrm}.fpsTRE = [trks{index_algrm}.fpsTRE results{index_results}.fps];
        
            end
        end
  
    end
end


%SRE FPS
pathRes = './results/results_SRE_CVPR13/';

for index_algrm=1:length(trks)
    trks{index_algrm}.fpsSRE = [];
end

for index_seq=1:length(seqs)
    seq = seqs{index_seq};
    seq_name = seq.name;
    
    for index_algrm=1:length(trks)
        algrm = trks{index_algrm};
        name=algrm.name;
        
        fileName = [pathRes seq_name '_' name '.mat'];
        load(fileName); 
        
        for index_results = 1:length(results)
            if length(trks{index_algrm}.fpsSRE) == 0 
                trks{index_algrm}.fpsSRE = results{index_results}.fps;
            else
                trks{index_algrm}.fpsSRE = [trks{index_algrm}.fpsSRE results{index_results}.fps];
        
            end
        end
  
    end
end

for index_algrm=1:length(trks)
   disp([trks{index_algrm}.namePaper ' ' num2str(mean(trks{index_algrm}.fpsSRE)) ' FPS']); 
end
