% NKG_make_sensorRMS average.m


temp = [];

for w = 1:numel(wordlist)
    
    thisword = char(wordlist(w));
    
    temp = [];
    
    if (~exist([rootCodeOutputPath, 'Version4_source_level_CUDA/', experimentName, '/1-sensor-data/fif-out-averaged-GRADS+MAGS-readyforNKG/gave/', thisword, '.fif'], 'file'))
        wordexists = 0;
        for p = 1:numel(participentIDlist)
            if exist([rootCodeOutputPath 'Version4_source_level_CUDA/' experimentName, '/1-sensor-data/fif-out-averaged/gave/', char(participentIDlist(p)), '-', thisword, '.fif'], 'file') && wordexists == 0
                sensordataveraged = fiff_read_evoked([rootCodeOutputPath 'Version4_source_level_CUDA/' experimentName, '/1-sensor-data/fif-out-averaged/gave/', char(participentIDlist(p)), '-', thisword, '.fif']);
            end
        end 
        
        
        
        for p = 1:numel(participentIDlist)
            
            
            
            if (exist([rootCodeOutputPath, 'Version4_source_level_CUDA/', experimentName, '/1-sensor-data/fif-out-averaged/gave/', char(participentIDlist(p)), '-', thisword, '.fif'], 'file'))
                
                sensordata = fiff_read_evoked([rootCodeOutputPath 'Version4_source_level_CUDA/' experimentName, '/1-sensor-data/fif-out-averaged/gave/', char(participentIDlist(p)), '-', thisword, '.fif']);
                
                % Average correct Mags and Grads
                
                for channel = 1:306
                    temp(channel,1:size(sensordata.evoked.epochs,2), p) = sensordata.evoked.epochs(find(strcmp(sensordataveraged.info.ch_names(channel), sensordata.info.ch_names)),1:size(sensordata.evoked.epochs,2));
                end
                
            else
                % Nothing
            end
        end
        
        
        %average over participants
        temp = mean(temp, 3);
        
        % Create RMS in every 3rd slot (Over Mags)
         for channel = 3:3:306
             temp(channel, :) = sqrt(temp(channel-2,:).^2 + temp(channel-1,:).^2);
        end
        
        
        %delete grads
        %for channel = 305:-3:1
        %    temp(channel, :) = [];
        %end
        %for channel = 203:-2:1
        %    temp(channel, :) = [];
        %end
        
        % Save
        sensordataveraged.evoked.epochs(1:306,:) = temp;
        
        fiff_write_evoked([rootCodeOutputPath 'Version4_source_level_CUDA/' experimentName, '/1-sensor-data/fif-out-averaged-GRADS+RMS-readyforNKG/gave/', thisword, '.fif'], sensordataveraged);
        
        
    end
    
end


